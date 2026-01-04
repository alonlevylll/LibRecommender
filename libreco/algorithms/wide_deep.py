"""Implementation of Wide & Deep."""
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


class WideDeep(TfBase, metaclass=ModelMeta):
    """*Wide & Deep* algorithm.

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
    lr : dict, default: {"wide": 0.01, "deep": 1e-4}
        Learning rate for training. The parameter should be a dict that contains
        learning rate of wide and deep parts.
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
    dense_wide_only : list of str or None, default: None
        List of dense feature column names that should only be used in the wide part.
        Features not in this list will be used in both wide and deep parts (unless
        specified in dense_deep_only).
        If None, no dense features are wide-only.
        
        Example: ``dense_wide_only=['age', 'income']`` means 'age' and 'income' go
        only to wide, while other dense features go to both wide and deep.
    dense_deep_only : list of str or None, default: None
        List of dense feature column names that should only be used in the deep part.
        Features not in this list will be used in both wide and deep parts (unless
        specified in dense_wide_only).
        If None, no dense features are deep-only.
        
        Example: ``dense_deep_only=['embedding_feat']`` means 'embedding_feat' goes
        only to deep, while other dense features go to both wide and deep.
        
        Note: A feature cannot be in both dense_wide_only and dense_deep_only.
    sparse_wide_only : list of str or None, default: None
        List of sparse feature column names that should only be used in the wide part.
        Features not in this list will be used in both wide and deep parts (unless
        specified in sparse_deep_only).
        If None, no sparse features are wide-only.
        
        Example: ``sparse_wide_only=['category']`` means 'category' goes
        only to wide, while other sparse features go to both wide and deep.
        
        Note: For multi-sparse features, if any column in a multi-sparse group is
        specified, the entire group will be treated together.
    sparse_deep_only : list of str or None, default: None
        List of sparse feature column names that should only be used in the deep part.
        Features not in this list will be used in both wide and deep parts (unless
        specified in sparse_wide_only).
        If None, no sparse features are deep-only.
        
        Example: ``sparse_deep_only=['tags']`` means 'tags' goes
        only to deep, while other sparse features go to both wide and deep.
        
        Note: For multi-sparse features, if any column in a multi-sparse group is
        specified, the entire group will be treated together.
        
        Note: A feature cannot be in both sparse_wide_only and sparse_deep_only.
    pos_sampler : {'random', 'unpopular'}, default: 'random'
        Positive example sampling strategy.

        - ``'random'`` means uniform random sampling of training examples.
        - ``'unpopular'`` oversamples low-volume items using weight = 1/sqrt(count),
          the same weighting scheme as WMSE loss. This helps the model learn better
          representations for items with few interactions.

    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple or None, default: None
        Lower and upper score bound for `rating` task.
    tf_sess_config : dict or None, default: None
        Optional TensorFlow session config, see `ConfigProto options
        <https://github.com/tensorflow/tensorflow/blob/v2.10.0/tensorflow/core/protobuf/config.proto#L431>`_.

    Notes
    -----
    According to the original paper, the Wide part uses FTRL with L1 regularization
    as the optimizer, so we'll also adopt it here. Note this may not be suitable for
    your specific task.

    References
    ----------
    *Heng-Tze Cheng et al.* `Wide & Deep Learning for Recommender Systems
    <https://arxiv.org/pdf/1606.07792.pdf>`_.

    """

    user_variables = ("embedding/user_wide_var", "embedding/user_deep_var")
    item_variables = ("embedding/item_wide_var", "embedding/item_deep_var")
    sparse_variables = ("embedding/sparse_wide_var", "embedding/sparse_deep_var", "embedding/sparse_wide_only_var", "embedding/sparse_deep_only_var")
    dense_variables = ("embedding/dense_wide_var", "embedding/dense_deep_var", "embedding/dense_wide_only_var", "embedding/dense_deep_only_var")

    def __init__(
        self,
        task,
        data_info=None,
        loss_type="cross_entropy",
        embed_size=16,
        n_epochs=20,
        lr=None,
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
        dense_wide_only=None,
        dense_deep_only=None,
        sparse_wide_only=None,
        sparse_deep_only=None,
        pos_sampler="random",
        seed=42,
        lower_upper_bound=None,
        tf_sess_config=None,
    ):
        super().__init__(task, data_info, lower_upper_bound, tf_sess_config)

        self.all_args = locals()
        self.loss_type = loss_type
        self.embed_size = embed_size
        self.n_epochs = n_epochs
        self.lr = self.check_lr(lr)
        self.lr_decay = lr_decay
        self.epsilon = epsilon
        self.reg = reg_config(reg)
        self.batch_size = batch_size
        self.sampler = sampler
        self.num_neg = num_neg
        self.use_bn = use_bn
        self.dropout_rate = dropout_config(dropout_rate)
        self.hidden_units = hidden_units_config(hidden_units)
        self.dense_wide_only = dense_wide_only
        self.dense_deep_only = dense_deep_only
        self.sparse_wide_only = sparse_wide_only
        self.sparse_deep_only = sparse_deep_only
        self.pos_sampler = pos_sampler
        self.seed = seed
        
        # Validate pos_sampler
        if pos_sampler not in ("random", "unpopular"):
            raise ValueError(
                f"`pos_sampler` must be one of ('random', 'unpopular'), got {pos_sampler}"
            )
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
        self.wide_embed, self.deep_embed = [], []

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()

        wide_embed = tf.concat(self.wide_embed, axis=1)
        wide_term = tf_dense(units=1, name="wide_term")(wide_embed)

        deep_embed = tf.concat(self.deep_embed, axis=1)
        deep_layer = dense_nn(
            deep_embed,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="deep",
        )
        deep_term = tf_dense(units=1, name="deep_term")(deep_layer)
        self.output = tf.squeeze(tf.add(wide_term, deep_term))
        self.serving_topk = self.build_topk(self.output)
        count_params()

    def _build_user_item(self):
        self.user_indices = tf.placeholder(tf.int32, shape=[None])
        self.item_indices = tf.placeholder(tf.int32, shape=[None])

        wide_user_embed = embedding_lookup(
            indices=self.user_indices,
            var_name="user_wide_var",
            var_shape=(self.n_users + 1, 1),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        wide_item_embed = embedding_lookup(
            indices=self.item_indices,
            var_name="item_wide_var",
            var_shape=(self.n_items + 1, 1),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        deep_user_embed = embedding_lookup(
            indices=self.user_indices,
            var_name="user_deep_var",
            var_shape=(self.n_users + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        deep_item_embed = embedding_lookup(
            indices=self.item_indices,
            var_name="item_deep_var",
            var_shape=(self.n_items + 1, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )

        self.wide_embed.extend([wide_user_embed, wide_item_embed])
        self.deep_embed.extend([deep_user_embed, deep_item_embed])

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
        
        # Create a new data_info-like object (we'll use a simple object to hold the info)
        # Actually, since compute_sparse_feats only uses multi_sparse_combine_info,
        # we can create a minimal wrapper
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
        
        # Split sparse indices into wide-only, deep-only, and both-network groups
        has_wide_only = len(self.sparse_wide_only_indices) > 0
        has_deep_only = len(self.sparse_deep_only_indices) > 0
        has_both = len(self.sparse_both_indices) > 0
        
        if has_wide_only:
            # Features that go only to wide network
            wide_only_indices = tf.constant(self.sparse_wide_only_indices, dtype=tf.int32)
            sparse_wide_only_indices_tensor = tf.gather(self.sparse_indices, wide_only_indices, axis=1)
            split_data_info_wide = self._create_split_data_info(self.sparse_wide_only_indices)
            
            wide_only_embed = compute_sparse_feats(
                split_data_info_wide,
                self.multi_sparse_combiner,
                sparse_wide_only_indices_tensor,
                var_name="sparse_wide_only_var",
                var_shape=[self.sparse_feature_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.wide_embed.append(wide_only_embed)
        
        if has_deep_only:
            # Features that go only to deep network
            deep_only_indices = tf.constant(self.sparse_deep_only_indices, dtype=tf.int32)
            sparse_deep_only_indices_tensor = tf.gather(self.sparse_indices, deep_only_indices, axis=1)
            split_data_info_deep = self._create_split_data_info(self.sparse_deep_only_indices)
            
            deep_only_embed = compute_sparse_feats(
                split_data_info_deep,
                self.multi_sparse_combiner,
                sparse_deep_only_indices_tensor,
                var_name="sparse_deep_only_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            self.deep_embed.append(deep_only_embed)
        
        if has_both:
            # Features that go to both wide and deep networks
            both_indices = tf.constant(self.sparse_both_indices, dtype=tf.int32)
            sparse_both_indices_tensor = tf.gather(self.sparse_indices, both_indices, axis=1)
            split_data_info_both = self._create_split_data_info(self.sparse_both_indices)
            
            wide_sparse_embed = compute_sparse_feats(
                split_data_info_both,
                self.multi_sparse_combiner,
                sparse_both_indices_tensor,
                var_name="sparse_wide_var",
                var_shape=[self.sparse_feature_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.wide_embed.append(wide_sparse_embed)
            
            deep_sparse_embed = compute_sparse_feats(
                split_data_info_both,
                self.multi_sparse_combiner,
                sparse_both_indices_tensor,
                var_name="sparse_deep_var",
                var_shape=(self.sparse_feature_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            self.deep_embed.append(deep_sparse_embed)

    def _setup_dense_split(self, data_info):
        """Setup indices for splitting dense features between wide-only, deep-only, and both networks."""
        dense_col_names = data_info.dense_col.name
        
        wide_only_set = set(self.dense_wide_only) if self.dense_wide_only else set()
        deep_only_set = set(self.dense_deep_only) if self.dense_deep_only else set()
        
        # Validate no overlap
        overlap = wide_only_set & deep_only_set
        if overlap:
            raise ValueError(
                f"Features cannot be in both dense_wide_only and dense_deep_only: {overlap}"
            )
        
        # Validate column names
        all_specified = wide_only_set | deep_only_set
        invalid_cols = all_specified - set(dense_col_names)
        if invalid_cols:
            raise ValueError(
                f"dense_wide_only/dense_deep_only contains invalid column names: {invalid_cols}. "
                f"Valid dense columns are: {dense_col_names}"
            )
        
        # Get indices for each group
        self.dense_wide_only_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name in wide_only_set
        ]
        self.dense_deep_only_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name in deep_only_set
        ]
        self.dense_both_indices = [
            i for i, name in enumerate(dense_col_names) 
            if name not in wide_only_set and name not in deep_only_set
        ]

    def _setup_sparse_split(self, data_info):
        """Setup indices for splitting sparse features between wide-only, deep-only, and both networks.
        
        Handles multi-sparse features by ensuring all columns in a multi-sparse group
        are treated together.
        """
        sparse_col_names = data_info.sparse_col.name
        
        wide_only_set = set(self.sparse_wide_only) if self.sparse_wide_only else set()
        deep_only_set = set(self.sparse_deep_only) if self.sparse_deep_only else set()
        
        # Validate no overlap
        overlap = wide_only_set & deep_only_set
        if overlap:
            raise ValueError(
                f"Features cannot be in both sparse_wide_only and sparse_deep_only: {overlap}"
            )
        
        # Validate column names
        all_specified = wide_only_set | deep_only_set
        invalid_cols = all_specified - set(sparse_col_names)
        if invalid_cols:
            raise ValueError(
                f"sparse_wide_only/sparse_deep_only contains invalid column names: {invalid_cols}. "
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
            
            # Expand wide_only_set and deep_only_set to include all columns in multi-sparse groups
            expanded_wide_only = set(wide_only_set)
            expanded_deep_only = set(deep_only_set)
            
            for col in list(wide_only_set):
                if col in col_to_main:
                    main_col = col_to_main[col]
                    # Find all columns in this multi-sparse group
                    for sparse_col in sparse_col_names:
                        if sparse_col == main_col or col_to_main.get(sparse_col) == main_col:
                            expanded_wide_only.add(sparse_col)
            
            for col in list(deep_only_set):
                if col in col_to_main:
                    main_col = col_to_main[col]
                    # Find all columns in this multi-sparse group
                    for sparse_col in sparse_col_names:
                        if sparse_col == main_col or col_to_main.get(sparse_col) == main_col:
                            expanded_deep_only.add(sparse_col)
            
            wide_only_set = expanded_wide_only
            deep_only_set = expanded_deep_only
        
        # Get field indices for each group
        self.sparse_wide_only_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name in wide_only_set
        ]
        self.sparse_deep_only_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name in deep_only_set
        ]
        self.sparse_both_indices = [
            i for i, name in enumerate(sparse_col_names) 
            if name not in wide_only_set and name not in deep_only_set
        ]

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        
        # Split dense values into wide-only, deep-only, and both-network groups
        has_wide_only = len(self.dense_wide_only_indices) > 0
        has_deep_only = len(self.dense_deep_only_indices) > 0
        has_both = len(self.dense_both_indices) > 0
        
        if has_wide_only:
            # Features that go only to wide network
            wide_only_indices = tf.constant(self.dense_wide_only_indices, dtype=tf.int32)
            dense_wide_only_values = tf.gather(self.dense_values, wide_only_indices, axis=1)
            wide_only_size = len(self.dense_wide_only_indices)
            
            wide_only_embed = compute_dense_feats(
                dense_wide_only_values,
                var_name="dense_wide_only_var",
                var_shape=[wide_only_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.wide_embed.append(wide_only_embed)
        
        if has_deep_only:
            # Features that go only to deep network
            deep_only_indices = tf.constant(self.dense_deep_only_indices, dtype=tf.int32)
            dense_deep_only_values = tf.gather(self.dense_values, deep_only_indices, axis=1)
            deep_only_size = len(self.dense_deep_only_indices)
            
            deep_only_embed = compute_dense_feats(
                dense_deep_only_values,
                var_name="dense_deep_only_var",
                var_shape=(deep_only_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            self.deep_embed.append(deep_only_embed)
        
        if has_both:
            # Features that go to both wide and deep networks
            both_indices = tf.constant(self.dense_both_indices, dtype=tf.int32)
            dense_both_values = tf.gather(self.dense_values, both_indices, axis=1)
            both_size = len(self.dense_both_indices)
            
            wide_dense_embed = compute_dense_feats(
                dense_both_values,
                var_name="dense_wide_var",
                var_shape=[both_size],
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
            )
            self.wide_embed.append(wide_dense_embed)
            
            deep_dense_embed = compute_dense_feats(
                dense_both_values,
                var_name="dense_deep_var",
                var_shape=(both_size, self.embed_size),
                initializer=tf.glorot_uniform_initializer(),
                regularizer=self.reg,
                flatten=True,
            )
            self.deep_embed.append(deep_dense_embed)

    @staticmethod
    def check_lr(lr):
        if not lr:
            return {"wide": 0.01, "deep": 1e-4}
        else:
            assert isinstance(lr, dict) and "wide" in lr and "deep" in lr, (
                "`lr` should be a dict that contains learning rate of "
                "wide and deep parts, e.g. {'wide': 0.01, 'deep': 1e-4}"
            )
            return lr
