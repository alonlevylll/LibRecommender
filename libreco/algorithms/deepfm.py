"""Implementation of DeepFM."""
import numpy as np

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
    sparse_linear_only : list of str or None, default: None
        List of sparse feature column names that should only be used in the linear part.
        Features not in this list will be used in linear, FM, and deep parts (unless
        specified in sparse_deep_only).
        If None, no sparse features are linear-only.
        
        Example: ``sparse_linear_only=['category']`` means 'category' goes
        only to linear, while other sparse features go to all parts.
        
        Note: For multi-sparse features, if any column in a multi-sparse group is
        specified, the entire group will be treated together.
    sparse_deep_only : list of str or None, default: None
        List of sparse feature column names that should only be used in the FM and deep parts
        (not linear). Features not in this list will be used in all parts (unless
        specified in sparse_linear_only).
        If None, no sparse features are deep-only.
        
        Example: ``sparse_deep_only=['tags']`` means 'tags' goes
        only to FM and deep, while other sparse features go to all parts.
        
        Note: For multi-sparse features, if any column in a multi-sparse group is
        specified, the entire group will be treated together.
        
        Note: A feature cannot be in both sparse_linear_only and sparse_deep_only.
    dense_linear_only : list of str or None, default: None
        List of dense feature column names that should only be used in the linear part.
        Features not in this list will be used in linear, FM, and deep parts (unless
        specified in dense_deep_only).
        If None, no dense features are linear-only.
        
        Example: ``dense_linear_only=['age', 'income']`` means 'age' and 'income' go
        only to linear, while other dense features go to all parts.
    dense_deep_only : list of str or None, default: None
        List of dense feature column names that should only be used in the FM and deep parts
        (not linear). Features not in this list will be used in all parts (unless
        specified in dense_linear_only).
        If None, no dense features are deep-only.
        
        Example: ``dense_deep_only=['embedding_feat']`` means 'embedding_feat' goes
        only to FM and deep, while other dense features go to all parts.
        
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
    sparse_variables = ("embedding/sparse_linear_var", "embedding/sparse_embeds_var", "embedding/sparse_linear_only_var", "embedding/sparse_deep_only_var")
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
        sparse_linear_only=None,
        sparse_deep_only=None,
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
        self.sparse_linear_only = sparse_linear_only
        self.sparse_deep_only = sparse_deep_only
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
            self._setup_sparse_split(data_info)
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

    def _setup_sparse_split(self, data_info):
        """Setup indices for splitting sparse features between linear-only, deep-only, and all parts.
        
        Handles multi-sparse features by ensuring all columns in a multi-sparse group
        are treated together.
        """
        sparse_col_names = data_info.sparse_col.name
        
        linear_only_set = set(self.sparse_linear_only) if self.sparse_linear_only else set()
        deep_only_set = set(self.sparse_deep_only) if self.sparse_deep_only else set()
        
        # Validate no overlap
        overlap = linear_only_set & deep_only_set
        if overlap:
            raise ValueError(
                f"Features cannot be in both sparse_linear_only and sparse_deep_only: {overlap}"
            )
        
        # Validate column names
        all_specified = linear_only_set | deep_only_set
        invalid_cols = all_specified - set(sparse_col_names)
        if invalid_cols:
            raise ValueError(
                f"sparse_linear_only/sparse_deep_only contains invalid column names: {invalid_cols}. "
                f"Valid sparse columns are: {sparse_col_names}"
            )
        
        # Handle multi-sparse: if any column in a multi-sparse group is specified,
        # include all columns in that group
        if data_info.multi_sparse_combine_info:
            multi_sparse_map = data_info.col_name_mapping.get("multi_sparse", {})
            # Map each column to its main multi-sparse column (if it's part of one)
            col_to_main = {}
            for sub_col, main_col in multi_sparse_map.items():
                col_to_main[sub_col] = main_col
                col_to_main[main_col] = main_col  # main column maps to itself
            
            # Expand linear_only_set and deep_only_set to include all columns in multi-sparse groups
            expanded_linear_only = set(linear_only_set)
            expanded_deep_only = set(deep_only_set)
            
            for col in list(linear_only_set):
                if col in col_to_main:
                    main_col = col_to_main[col]
                    # Find all columns in this multi-sparse group
                    for sparse_col in sparse_col_names:
                        if sparse_col == main_col or col_to_main.get(sparse_col) == main_col:
                            expanded_linear_only.add(sparse_col)
            
            for col in list(deep_only_set):
                if col in col_to_main:
                    main_col = col_to_main[col]
                    # Find all columns in this multi-sparse group
                    for sparse_col in sparse_col_names:
                        if sparse_col == main_col or col_to_main.get(sparse_col) == main_col:
                            expanded_deep_only.add(sparse_col)
            
            linear_only_set = expanded_linear_only
            deep_only_set = expanded_deep_only
        
        # Get field indices for each group
        self.sparse_linear_only_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name in linear_only_set
        ]
        self.sparse_deep_only_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name in deep_only_set
        ]
        self.sparse_all_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name not in linear_only_set and name not in deep_only_set
        ]

    def _create_split_data_info(self, field_indices):
        """Create a modified data_info structure for a subset of sparse fields.
        
        Adjusts multi_sparse_combine_info field offsets to match the sliced indices.
        """
        if not field_indices:
            return None
        
        # Create a mapping from old field index to new field index
        field_set = set(field_indices)
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(field_indices))}
        
        # Check if we need to adjust multi_sparse_combine_info
        if not self.data_info.multi_sparse_combine_info:
            # No multi-sparse, simple case - can use original data_info
            return self.data_info
        
        # Adjust multi_sparse_combine_info
        original_field_offsets = self.data_info.multi_sparse_combine_info.field_offset
        original_field_lens = self.data_info.multi_sparse_combine_info.field_len
        original_feat_oovs = self.data_info.multi_sparse_combine_info.feat_oov
        
        new_field_offsets = []
        new_field_lens = []
        new_feat_oovs = []
        
        # Check each multi-sparse field
        for field_idx, (offset, length, oov) in enumerate(
            zip(original_field_offsets, original_field_lens, original_feat_oovs)
        ):
            # Check if all columns in this multi-sparse field are in field_indices
            field_start = offset
            field_end = offset + length
            field_col_indices = list(range(field_start, field_end))
            
            if all(col_idx in field_set for col_idx in field_col_indices):
                # All columns present, adjust offset
                new_offset = old_to_new[field_start]
                new_field_offsets.append(new_offset)
                new_field_lens.append(length)
                new_feat_oovs.append(oov)
        
        # Create a new data_info-like object
        class SplitDataInfo:
            def __init__(self, original_data_info, multi_sparse_info):
                self.multi_sparse_combine_info = multi_sparse_info
                # Copy other attributes we might need
                self.sparse_unique_vals = original_data_info.sparse_unique_vals
                self.multi_sparse_unique_vals = original_data_info.multi_sparse_unique_vals
                self.col_name_mapping = original_data_info.col_name_mapping
        
        if new_field_offsets:
            from ..data.data_info import MultiSparseInfo
            new_multi_sparse_info = MultiSparseInfo(
                new_field_offsets,
                new_field_lens,
                np.array(new_feat_oovs),
                self.data_info.multi_sparse_combine_info.pad_val
            )
            return SplitDataInfo(self.data_info, new_multi_sparse_info)
        else:
            # No multi-sparse fields in this split
            return SplitDataInfo(self.data_info, None)

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        
        # Split sparse indices into linear-only, deep-only, and all-parts groups
        has_linear_only = len(self.sparse_linear_only_indices) > 0
        has_deep_only = len(self.sparse_deep_only_indices) > 0
        has_all = len(self.sparse_all_indices) > 0
        
        if has_linear_only:
            # Features that go only to linear part
            linear_only_indices = tf.constant(self.sparse_linear_only_indices, dtype=tf.int32)
            sparse_linear_only_tensor = tf.gather(self.sparse_indices, linear_only_indices, axis=1)
            split_data_info_linear = self._create_split_data_info(self.sparse_linear_only_indices)
            
            linear_only_embed = compute_sparse_feats(
                split_data_info_linear,
                self.multi_sparse_combiner,
                sparse_linear_only_tensor,
                var_name="sparse_linear_only_var",
                var_shape=[self.sparse_feature_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.linear_embed.append(linear_only_embed)
        
        if has_deep_only:
            # Features that go only to FM (pairwise) and deep parts (not linear)
            deep_only_indices = tf.constant(self.sparse_deep_only_indices, dtype=tf.int32)
            sparse_deep_only_tensor = tf.gather(self.sparse_indices, deep_only_indices, axis=1)
            split_data_info_deep = self._create_split_data_info(self.sparse_deep_only_indices)
            
            pairwise_deep_only_embed = compute_sparse_feats(
                split_data_info_deep,
                self.multi_sparse_combiner,
                sparse_deep_only_tensor,
                var_name="sparse_deep_only_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            deep_only_embed = tf.keras.layers.Flatten()(pairwise_deep_only_embed)
            
            self.pairwise_embed.append(pairwise_deep_only_embed)
            self.deep_embed.append(deep_only_embed)
        
        if has_all:
            # Features that go to all parts (linear, FM, deep)
            all_indices = tf.constant(self.sparse_all_indices, dtype=tf.int32)
            sparse_all_tensor = tf.gather(self.sparse_indices, all_indices, axis=1)
            split_data_info_all = self._create_split_data_info(self.sparse_all_indices)
            
            linear_sparse_embed = compute_sparse_feats(
                split_data_info_all,
                self.multi_sparse_combiner,
                sparse_all_tensor,
                var_name="sparse_linear_var",
                var_shape=[self.sparse_feature_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            pairwise_sparse_embed = compute_sparse_feats(
                split_data_info_all,
                self.multi_sparse_combiner,
                sparse_all_tensor,
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
