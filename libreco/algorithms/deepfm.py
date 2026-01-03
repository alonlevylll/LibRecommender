"""Implementation of DeepFM."""
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


class DeepFM(TfBase, metaclass=ModelMeta):
    """*DeepFM* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'wmse'}, default: 'cross_entropy'
        Loss for model training. For rating task, 'wmse' (Weighted Mean Squared Error)
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

        .. versionchanged:: 1.0.0
           Accept type of ``int``, ``list`` or ``tuple``, instead of ``str``.

    multi_sparse_combiner : {'normal', 'mean', 'sum', 'sqrtn'}, default: 'sqrtn'
        Options for combining `multi_sparse` features.
    dense_linear_only : list of str or None, default: None
        List of dense feature column names that should only be used in the linear part.
        Features not in this list will be used in linear, CIN, and deep parts (unless
        specified in dense_deep_only).
        If None, no dense features are linear-only.
        
        Example: ``dense_linear_only=['age', 'income']`` means 'age' and 'income' go
        only to linear, while other dense features go to all parts.
    dense_deep_only : list of str or None, default: None
        List of dense feature column names that should only be used in the deep and CIN parts
        (not linear). Features not in this list will be used in all parts (unless
        specified in dense_linear_only).
        If None, no dense features are deep-only.
        
        Example: ``dense_deep_only=['embedding_feat']`` means 'embedding_feat' goes
        only to CIN and deep, while other dense features go to all parts.
        
        Note: A feature cannot be in both dense_linear_only and dense_deep_only.
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    References
    ----------
    *Huifeng Guo et al.* `DeepFM: A Factorization-Machine based Neural Network for CTR Prediction
    <https://arxiv.org/pdf/1703.04247.pdf>`_.
    """

    user_variables = ("embedding/user_linear_var", "embedding/user_embeds_var")
    item_variables = ("embedding/item_linear_var", "embedding/item_embeds_var")
    sparse_variables = ("embedding/sparse_linear_var", "embedding/sparse_embeds_var")
    dense_variables = ("embedding/dense_linear_var", "embedding/dense_embeds_var", "embedding/dense_linear_only_var", "embedding/dense_deep_only_var")

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
        multi_sparse_combiner="sqrtn",
        dense_linear_only=None,
        dense_deep_only=None,
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
        self.dense_linear_only = dense_linear_only
        self.dense_deep_only = dense_deep_only
        self.seed = seed
        self.sparse = check_sparse_indices(data_info)
        self.dense = check_dense_values(data_info)
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
            self._setup_dense_split(data_info)

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.labels = tf.placeholder(tf.float32, shape=[None])
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.linear_embed, self.pairwise_embed, self.deep_embed = [], [], []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        linear_embed = tf.concat(self.linear_embed, axis=1)
        pairwise_embed = tf.concat(self.pairwise_embed, axis=1)
        deep_embed = tf.concat(self.deep_embed, axis=1)

        linear_term = tf_dense(units=1, activation=None)(linear_embed)
        pairwise_term = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(pairwise_embed, axis=1)),
            tf.reduce_sum(tf.square(pairwise_embed), axis=1),
        )
        deep_term = dense_nn(
            deep_embed,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="deep",
        )

        concat_layer = tf.concat([linear_term, pairwise_term, deep_term], axis=1)
        self.output = tf.squeeze(tf_dense(units=1, activation=None)(concat_layer))
        self.serving_topk = self.build_topk(self.output)
        count_params()

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
        self.pairwise_embed.extend(
            [user_embeds[:, tf.newaxis, :], item_embeds[:, tf.newaxis, :]]
        )
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
        deep_sparse_embed = tf.keras.layers.Flatten()(pairwise_sparse_embed)
        self.linear_embed.append(linear_sparse_embed)
        self.pairwise_embed.append(pairwise_sparse_embed)
        self.deep_embed.append(deep_sparse_embed)

    def _setup_dense_split(self, data_info):
        """Setup indices for splitting dense features between linear-only, deep-only, and all parts."""
        dense_col_names = data_info.dense_col.name
        
        linear_only_set = set(self.dense_linear_only) if self.dense_linear_only else set()
        deep_only_set = set(self.dense_deep_only) if self.dense_deep_only else set()
        
        # Validate no overlap
        overlap = linear_only_set & deep_only_set
        if overlap:
            raise ValueError(
                f"Features cannot be in both dense_linear_only and dense_deep_only: {overlap}"
            )
        
        # Validate column names
        all_specified = linear_only_set | deep_only_set
        invalid_cols = all_specified - set(dense_col_names)
        if invalid_cols:
            raise ValueError(
                f"dense_linear_only/dense_deep_only contains invalid column names: {invalid_cols}. "
                f"Valid dense columns are: {dense_col_names}"
            )
        
        # Get indices for each group
        self.dense_linear_only_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name in linear_only_set
        ]
        self.dense_deep_only_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name in deep_only_set
        ]
        self.dense_all_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name not in linear_only_set and name not in deep_only_set
        ]

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        
        # Split dense values into linear-only, deep-only, and all-parts groups
        has_linear_only = len(self.dense_linear_only_indices) > 0
        has_deep_only = len(self.dense_deep_only_indices) > 0
        has_all = len(self.dense_all_indices) > 0
        
        if has_linear_only:
            # Features that go only to linear part
            linear_only_indices = tf.constant(self.dense_linear_only_indices, dtype=tf.int32)
            dense_linear_only_values = tf.gather(self.dense_values, linear_only_indices, axis=1)
            linear_only_size = len(self.dense_linear_only_indices)
            
            linear_only_embed = compute_dense_feats(
                dense_linear_only_values,
                var_name="dense_linear_only_var",
                var_shape=[linear_only_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.linear_embed.append(linear_only_embed)
        
        if has_deep_only:
            # Features that go only to pairwise and deep parts (not linear)
            deep_only_indices = tf.constant(self.dense_deep_only_indices, dtype=tf.int32)
            dense_deep_only_values = tf.gather(self.dense_values, deep_only_indices, axis=1)
            deep_only_size = len(self.dense_deep_only_indices)
            
            pairwise_deep_only_embed = compute_dense_feats(
                dense_deep_only_values,
                var_name="dense_deep_only_var",
                var_shape=(deep_only_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            
            # For deep: flatten
            deep_only_deep_embed = tf.keras.layers.Flatten()(pairwise_deep_only_embed)
            
            self.pairwise_embed.append(pairwise_deep_only_embed)
            self.deep_embed.append(deep_only_deep_embed)
        
        if has_all:
            # Features that go to all parts (linear, pairwise, deep)
            all_indices = tf.constant(self.dense_all_indices, dtype=tf.int32)
            dense_all_values = tf.gather(self.dense_values, all_indices, axis=1)
            all_size = len(self.dense_all_indices)
            
            linear_dense_embed = compute_dense_feats(
                dense_all_values,
                var_name="dense_linear_var",
                var_shape=[all_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            pairwise_dense_embed = compute_dense_feats(
                dense_all_values,
                var_name="dense_embeds_var",
                var_shape=(all_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            
            # For deep: flatten
            deep_dense_embed = tf.keras.layers.Flatten()(pairwise_dense_embed)
            
            self.linear_embed.append(linear_dense_embed)
            self.pairwise_embed.append(pairwise_dense_embed)
            self.deep_embed.append(deep_dense_embed)
