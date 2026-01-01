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
        Features not in this list will be used in both wide and deep parts.
        If None, all dense features are used in both parts.
        
        Example: ``dense_wide_only=['age', 'income']`` means 'age' and 'income' go
        only to wide, while other dense features go to both wide and deep.
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
    sparse_variables = ("embedding/sparse_wide_var", "embedding/sparse_deep_var")
    dense_variables = ("embedding/dense_wide_var", "embedding/dense_deep_var", "embedding/dense_wide_only_var")

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

    def _build_sparse(self):
        self.sparse_indices = tf.placeholder(
            tf.int32, shape=[None, self.sparse_field_size]
        )
        wide_sparse_embed = compute_sparse_feats(
            self.data_info,
            self.multi_sparse_combiner,
            self.sparse_indices,
            var_name="sparse_wide_var",
            var_shape=[self.sparse_feature_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        deep_sparse_embed = compute_sparse_feats(
            self.data_info,
            self.multi_sparse_combiner,
            self.sparse_indices,
            var_name="sparse_deep_var",
            var_shape=(self.sparse_feature_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            flatten=True,
        )
        self.wide_embed.append(wide_sparse_embed)
        self.deep_embed.append(deep_sparse_embed)

    def _setup_dense_split(self, data_info):
        """Setup indices for splitting dense features between wide-only and both networks."""
        dense_col_names = data_info.dense_col.name
        
        if self.dense_wide_only is None:
            # All features go to both networks
            self.dense_wide_only_indices = []
            self.dense_both_indices = list(range(len(dense_col_names)))
        else:
            # Validate column names
            invalid_cols = set(self.dense_wide_only) - set(dense_col_names)
            if invalid_cols:
                raise ValueError(
                    f"dense_wide_only contains invalid column names: {invalid_cols}. "
                    f"Valid dense columns are: {dense_col_names}"
                )
            
            # Get indices for each group
            self.dense_wide_only_indices = [
                i for i, name in enumerate(dense_col_names) 
                if name in self.dense_wide_only
            ]
            self.dense_both_indices = [
                i for i, name in enumerate(dense_col_names) 
                if name not in self.dense_wide_only
            ]

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        
        # Split dense values into wide-only and both-network groups
        has_wide_only = len(self.dense_wide_only_indices) > 0
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
