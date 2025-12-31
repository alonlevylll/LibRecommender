"""Implementation of xDeepFM."""
from ..bases import ModelMeta, TfBase
from ..feature.multi_sparse import true_sparse_field_size
from ..layers import dense_nn, embedding_lookup, tf_dense
from ..tfops import dropout_config, reg_config, tf
from ..tfops.features import compute_dense_feats, compute_sparse_feats
from ..torchops import hidden_units_config
from ..utils.misc import count_params
from ..utils.validate import (
    check_dense_values,
    check_multi_sparse,
    check_sparse_indices,
    dense_field_size,
    sparse_feat_size,
    sparse_field_size,
)


class xDeepFM(TfBase, metaclass=ModelMeta):
    """*xDeepFM* algorithm.

    xDeepFM combines a CIN (Compressed Interaction Network) with a classical DNN.
    The model is able to learn certain bounded-degree feature interactions explicitly
    through CIN; Besides, it can also learn arbitrary low- and high-order feature
    interactions implicitly through DNN.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'wrmse'}, default: 'cross_entropy'
        Loss for model training. For rating task, 'wrmse' (Weighted Root Mean Squared Error)
        can be used to weight items by their frequency in the training data.
    embed_size: int, default: 16
        Vector size of embeddings.
    n_epochs: int, default: 10
        Number of epochs for training.
    lr : float, default 0.001
        Learning rate for training.
    lr_decay : bool, default: False
        Whether to use learning rate decay.
    epsilon : float, default: 1e-5
        A small constant added to the denominator to improve numerical stability in
        Adam optimizer.
        According to the `official comment <https://github.com/tensorflow/tensorflow/blob/v1.15.0/tensorflow/python/training/adam.py#L64>`_,
        default value of `1e-8` for `epsilon` is generally not good, so here we choose `1e-5`.
        Users can try tuning this hyperparameter if the training is unstable.
    reg : float or None, default: None
        Regularization parameter, must be non-negative or None.
    batch_size : int, default: 256
        Batch size for training.
    sampler : {'random', 'unconsumed', 'popular'}, default: 'random'
        Negative sampling strategy.

        - ``'random'`` means random sampling.
        - ``'unconsumed'`` samples items that the target user did not consume before.
        - ``'popular'`` has a higher probability to sample popular items as negative samples.

        .. versionadded:: 1.1.0

    num_neg : int, default: 1
        Number of negative samples for each positive sample, only used in `ranking` task.
    use_bn : bool, default: True
        Whether to use batch normalization.
    dropout_rate : float or None, default: None
        Probability of an element to be zeroed. If it is None, dropout is not used.
    hidden_units : int, list of int or tuple of (int,), default: (128, 64, 32)
        Number of layers and corresponding layer size in MLP.
    cin_layer_size : list of int, default: [50, 50]
        Size of each CIN layer. Each element represents the number of feature maps
        in that layer. Larger sizes enable learning higher-order vector-wise interactions.
    cin_direct : bool, default: True
        Whether to use direct connections in CIN. If True, all feature maps are kept.
        If False, feature maps are split in half (except the last layer).
    multi_sparse_combiner : {'normal', 'mean', 'sum', 'sqrtn'}, default: 'sqrtn'
        Options for combining `multi_sparse` features.
    masked_data : bool, default: False
        Whether to use masked user-item ratings as additional features. When enabled,
        the model embeds all users' ratings for all items, but masks out the current
        item being predicted during training to avoid data leakage.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    *Jianxun Lian et al.* `xDeepFM: Combining Explicit and Implicit Feature Interactions
    for Recommender Systems <https://arxiv.org/pdf/1803.05170.pdf>`_.
    """

    user_variables = ("embedding/user_linear_var", "embedding/user_embeds_var")
    item_variables = ("embedding/item_linear_var", "embedding/item_embeds_var")
    sparse_variables = ("embedding/sparse_linear_var", "embedding/sparse_embeds_var")
    dense_variables = ("embedding/dense_linear_var", "embedding/dense_embeds_var")

    def __init__(
        self,
        task,
        data_info,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=0.001,
        lr_decay=False,
        epsilon=1e-5,
        reg=None,
        batch_size=256,
        sampler="random",
        num_neg=1,
        use_bn=True,
        dropout_rate=None,
        hidden_units=(128, 64, 32),
        cin_layer_size=(50, 50),
        cin_direct=True,
        multi_sparse_combiner="sqrtn",
        masked_data=False,
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.loss_type = loss_type
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = hidden_units_config(hidden_units)
        self.cin_layer_size = list(cin_layer_size) if isinstance(cin_layer_size, (list, tuple)) else [cin_layer_size]
        self.cin_direct = cin_direct
        self.masked_data = masked_data
        self.seed = seed
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
        
        # Adjust CIN layer sizes if not using direct connections
        if not self.cin_direct:
            self.cin_layer_size = [int(x // 2 * 2) for x in self.cin_layer_size]
        
        if self.sparse:
            self.sparse_feature_size = sparse_feat_size(data_info)
            self.sparse_field_size = sparse_field_size(data_info)
            self.multi_sparse_combiner = check_multi_sparse(
                data_info, multi_sparse_combiner
            )
            self.true_sparse_field_size = true_sparse_field_size(
                data_info, self.sparse_field_size, self.multi_sparse_combiner
            )
        if self.dense:
            self.dense_field_size = dense_field_size(data_info)

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.linear_embed, self.cin_embed, self.deep_embed = [], [], []

        self._build_user_item()
        if self.masked_data:
            self._build_masked_ratings()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        # Linear term
        linear_embed = tf.concat(self.linear_embed, axis=1)
        linear_term = tf_dense(units=1, activation=None, name="linear_term")(linear_embed)

        # CIN term
        cin_input = tf.concat(self.cin_embed, axis=1)  # [batch_size, field_num, embed_size]
        cin_output = self._compressed_interaction_network(cin_input)
        cin_term = tf_dense(units=1, activation=None, name="cin_term")(cin_output)

        # Deep term
        deep_embed = tf.concat(self.deep_embed, axis=1)
        deep_term = dense_nn(
            deep_embed,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="deep",
        )
        deep_term = tf_dense(units=1, activation=None, name="deep_term")(deep_term)

        # Combine all terms
        concat_layer = tf.concat([linear_term, cin_term, deep_term], axis=1)
        self.output = tf.squeeze(tf_dense(units=1, activation=None, name="output")(concat_layer))
        self.serving_topk = self.build_topk(self.output)
        count_params()

    def _compressed_interaction_network(self, input_features):
        """Compressed Interaction Network for explicit vector-wise feature interactions.
        
        For k-th CIN layer, the output X_k is calculated via:
        x_{h,*}^{k} = sum_{i=1}^{H_{k-1}} sum_{j=1}^{m} W_{i,j}^{k,h} (X_{i,*}^{k-1} \circ x_{j,*}^0)
        
        where H_k is the number of feature vectors in the k-th layer,
        and \circ denotes the Hadamard product.
        
        Args:
            input_features: Tensor of shape [batch_size, field_num, embed_size]
            
        Returns:
            Tensor of shape [batch_size, final_len] where final_len is the sum of
            all CIN layer sizes (or half for non-direct connections)
        """
        batch_size = tf.shape(input_features)[0]
        field_num = input_features.get_shape().as_list()[1]
        embed_size = input_features.get_shape().as_list()[2]
        
        # Track field numbers for each layer
        field_nums = [field_num]
        hidden_layers = [input_features]
        final_results = []
        
        # Create Conv1D layers for each CIN layer
        with tf.variable_scope("cin", reuse=tf.AUTO_REUSE):
            for i, layer_size in enumerate(self.cin_layer_size):
                # Compute outer product: z_i = X_{k-1} \circ X_0
                # Using einsum: "bhd,bmd->bhmd" 
                # hidden_layers[-1]: [batch_size, H_{k-1}, embed_size]
                # hidden_layers[0]: [batch_size, m, embed_size] where m=field_num
                # Result: [batch_size, H_{k-1}, m, embed_size]
                z_i = tf.einsum(
                    "bhd,bmd->bhmd",
                    hidden_layers[-1],
                    hidden_layers[0],
                )
                
                # Reshape for Conv1D: [batch_size, H_{k-1} * m, embed_size]
                z_i_reshaped = tf.reshape(
                    z_i,
                    [batch_size, field_nums[i] * field_nums[0], embed_size]
                )
                
                # Apply Conv1D with kernel_size=1
                # Input: [batch_size, H_{k-1} * m, embed_size]
                # Output: [batch_size, H_{k-1} * m, layer_size]
                z_i_conv = tf.layers.conv1d(
                    inputs=z_i_reshaped,
                    filters=layer_size,
                    kernel_size=1,
                    kernel_initializer=tf.glorot_uniform_initializer(),
                    kernel_regularizer=self.reg,
                    name=f"cin_conv_{i}",
                )
                
                # Apply activation
                output = tf.nn.relu(z_i_conv)  # [batch_size, H_{k-1} * m, layer_size]
                
                # Handle direct vs split connections
                if self.cin_direct:
                    direct_connect = output  # [batch_size, H_{k-1} * m, layer_size]
                    next_hidden = output
                    field_nums.append(layer_size)
                else:
                    if i != len(self.cin_layer_size) - 1:
                        # Split in half along the filter dimension
                        split_size = layer_size // 2
                        direct_connect, next_hidden = tf.split(
                            output, [split_size, split_size], axis=2
                        )
                        field_nums.append(split_size)
                    else:
                        # Last layer: use all
                        direct_connect = output
                        next_hidden = None
                        field_nums.append(layer_size)
                
                # Sum pooling over the H_{k-1} * m dimension
                # direct_connect: [batch_size, H_{k-1} * m, layer_size]
                # Sum over axis=1: [batch_size, layer_size]
                pooled = tf.reduce_sum(direct_connect, axis=1)
                final_results.append(pooled)
                
                if next_hidden is not None:
                    # Reshape next_hidden for next iteration
                    # next_hidden: [batch_size, H_{k-1} * m, H_k]
                    # We need to reshape to [batch_size, H_k, embed_size]
                    # This requires projecting back to embed_size dimension
                    # Actually, we should maintain the structure: [batch_size, H_k, embed_size]
                    # But Conv1D output is [batch_size, H_{k-1} * m, H_k]
                    # We need to reshape to [batch_size, H_k, embed_size]
                    # Option: use another Conv1D to project, or reshape differently
                    # Looking at PyTorch code, it seems next_hidden should be [batch_size, H_k, embed_size]
                    # So we need to project the [batch_size, H_{k-1} * m, H_k] to [batch_size, H_k, embed_size]
                    # Actually, let's transpose and use another conv to get back to embed_size
                    next_hidden_transposed = tf.transpose(next_hidden, [0, 2, 1])  # [batch_size, H_k, H_{k-1} * m]
                    # Project to embed_size
                    next_hidden_proj = tf.layers.conv1d(
                        inputs=next_hidden_transposed,
                        filters=embed_size,
                        kernel_size=1,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        kernel_regularizer=self.reg,
                        name=f"cin_proj_{i}",
                    )  # [batch_size, H_k, embed_size]
                    hidden_layers.append(next_hidden_proj)
        
        # Concatenate all pooled results
        result = tf.concat(final_results, axis=1)  # [batch_size, final_len]
        return result

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        linear_user_embeds = embedding_lookup(
            indices=self.user_indices,
            var_name="user_linear_var",
            var_shape=(self.n_users + 1, 1),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        linear_item_embeds = embedding_lookup(
            indices=self.item_indices,
            var_name="item_linear_var",
            var_shape=(self.n_items + 1, 1),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        user_embeds = embedding_lookup(
            indices=self.user_indices,
            var_name="user_embeds_var",
            var_shape=(self.n_users + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        item_embeds = embedding_lookup(
            indices=self.item_indices,
            var_name="item_embeds_var",
            var_shape=(self.n_items + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        self.linear_embed.extend([linear_user_embeds, linear_item_embeds])
        self.cin_embed.extend([user_embeds[:, tf.newaxis, :], item_embeds[:, tf.newaxis, :]])
        self.deep_embed.extend([user_embeds, item_embeds])

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        linear_sparse_embed = compute_sparse_feats(
            self.data_info,
            self.multi_sparse_combiner,
            self.sparse_indices,
            var_name="sparse_linear_var",
            var_shape=[self.sparse_feature_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        pairwise_sparse_embed = compute_sparse_feats(
            self.data_info,
            self.multi_sparse_combiner,
            self.sparse_indices,
            var_name="sparse_embeds_var",
            var_shape=(self.sparse_feature_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        # Reshape for CIN: [batch_size, field_num, embed_size]
        # For sparse features, we need to handle multi-sparse case
        if self.multi_sparse_combiner:
            # Multi-sparse features are already combined, reshape appropriately
            sparse_cin_embed = pairwise_sparse_embed
            if len(sparse_cin_embed.get_shape()) == 2:
                sparse_cin_embed = sparse_cin_embed[:, tf.newaxis, :]
        else:
            sparse_cin_embed = pairwise_sparse_embed
            if len(sparse_cin_embed.get_shape()) == 2:
                sparse_cin_embed = sparse_cin_embed[:, tf.newaxis, :]
        
        deep_sparse_embed = tf.keras.layers.Flatten()(pairwise_sparse_embed)
        self.linear_embed.append(linear_sparse_embed)
        self.cin_embed.append(sparse_cin_embed)
        self.deep_embed.append(deep_sparse_embed)

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        linear_dense_embed = compute_dense_feats(
            self.dense_values,
            var_name="dense_linear_var",
            var_shape=[self.dense_field_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        pairwise_dense_embed = compute_dense_feats(
            self.dense_values,
            var_name="dense_embeds_var",
            var_shape=(self.dense_field_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        # Reshape for CIN: [batch_size, field_num, embed_size]
        dense_cin_embed = pairwise_dense_embed
        if len(dense_cin_embed.get_shape()) == 2:
            dense_cin_embed = dense_cin_embed[:, tf.newaxis, :]
        
        deep_dense_embed = tf.keras.layers.Flatten()(pairwise_dense_embed)
        self.linear_embed.append(linear_dense_embed)
        self.cin_embed.append(dense_cin_embed)
        self.deep_embed.append(deep_dense_embed)

    def _build_masked_ratings(self):
        """Build masked user-item ratings embeddings.
        
        For rating tasks, this uses actual rating values (floating point numbers)
        as dense features, not one-hot encodings. The ratings matrix contains
        all user ratings for all items, and the current item's rating is masked
        out (set to zero) during training to avoid data leakage.
        """
        # Placeholder for user-item ratings matrix [n_users, n_items]
        # Contains actual rating values (floats), not binary indicators
        self.user_item_ratings = tf.placeholder(
            tf.float32, shape=[self.n_users, self.n_items], name="user_item_ratings"
        )
        
        # Gather ratings for users in batch [batch_size, n_items]
        # Each row is a dense vector of rating values for all items
        user_ratings = tf.gather(self.user_item_ratings, self.user_indices)
        
        # Create mask: set current item ratings to zero during training
        # Use one-hot encoding only to create the mask (not for ratings themselves)
        item_one_hot = tf.one_hot(self.item_indices, self.n_items, dtype=tf.float32)
        # Mask: zeros at current item positions, ones elsewhere
        mask = 1.0 - item_one_hot
        
        # Apply mask: zero out current item ratings (keep actual rating values for other items)
        masked_ratings = user_ratings * mask
        
        # Embed masked ratings for linear part (1D embedding)
        linear_ratings_embed = tf_dense(units=1, name="ratings_linear")(masked_ratings)
        linear_ratings_embed = tf.reduce_sum(linear_ratings_embed, axis=1, keepdims=True)
        
        # Embed masked ratings for CIN part (embed_size embedding, reshaped for CIN)
        cin_ratings_embed = tf_dense(units=self.embed_size, name="ratings_cin")(masked_ratings)
        # Reshape to [batch_size, n_items, embed_size] for CIN
        cin_ratings_embed = tf.reshape(
            cin_ratings_embed, [-1, self.n_items, self.embed_size]
        )
        
        # Embed masked ratings for deep part (embed_size embedding)
        deep_ratings_embed = tf_dense(units=self.embed_size, name="ratings_deep")(masked_ratings)
        deep_ratings_embed = tf.reduce_sum(deep_ratings_embed, axis=1)
        
        self.linear_embed.append(linear_ratings_embed)
        self.cin_embed.append(cin_ratings_embed)
        self.deep_embed.append(deep_ratings_embed)

