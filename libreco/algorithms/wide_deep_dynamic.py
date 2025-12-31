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


class WideDeepDynamic(TfBase, metaclass=ModelMeta):
    """*Wide & Deep Dynamic* algorithm.

    Parameters
    ----------
    task : {'rating', 'ranking'}
        Recommendation task. See :ref:`Task`.
    data_info : :class:`~libreco.data.DataInfo` object
        Object that contains useful information for training and inference.
    loss_type : {'cross_entropy', 'focal', 'wmse', 'softmax'}, default: 'cross_entropy'
        Loss for model training. For rating task:

        - 'wmse' (Weighted Mean Squared Error) can be used to weight items by
          their frequency in the training data.
        - 'softmax' treats rating prediction as a multi-class classification problem.
          The output layer will have N neurons (one per rating class), and softmax
          is applied to get probabilities. The final prediction is a weighted average
          of class probabilities. Rating labels are automatically extracted from
          the unique values in the training data labels.

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
    use_user_rating_vector : bool or str, default: False
        Whether to use user's rating history as an interaction-based dense feature.
        During training, the rating for the current item is masked (set to 0) to prevent
        data leakage. During inference, the full rating vector is used.
        
        - ``False``: Disabled (default)
        - ``True`` or ``'both'``: Use in both wide and deep parts
        - ``'wide'``: Use only in wide part (simpler, less prone to overfitting)
        - ``'deep'``: Use only in deep part
    use_user_rating_stats : bool or str, default: False
        Whether to use user's rating statistics (mean, std) as an interaction-based 
        dense feature. During training, the statistics are computed excluding the 
        current item's rating to prevent data leakage. During inference, full stats are used.
        
        - ``False``: Disabled (default)
        - ``True`` or ``'both'``: Use in both wide and deep parts
        - ``'wide'``: Use only in wide part
        - ``'deep'``: Use only in deep part
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
    dense_variables = ("embedding/dense_wide_var", "embedding/dense_deep_var")
    rating_vector_variables = ("embedding/rating_vector_wide_var", "embedding/rating_vector_deep_var")
    rating_stats_variables = ("embedding/rating_stats_wide_var", "embedding/rating_stats_deep_var")

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
        use_user_rating_vector=False,
        use_user_rating_stats=False,
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
        self.use_user_rating_vector = use_user_rating_vector
        self.use_user_rating_stats = use_user_rating_stats
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

        # Softmax loss validation (rating labels will be extracted from train_data in fit())
        if loss_type == "softmax" and task != "rating":
            raise ValueError("Softmax loss is only supported for rating task.")

        # Will be set in fit() when loss_type="softmax"
        self.rating_labels = None
        self.n_rating_classes = None
        self.rating_label_to_index = None
        
        # Store sparse_interaction for user rating vector feature
        self.sparse_interaction = None

    def build_model(self):
        tf.set_random_seed(self.seed)
        self.is_training = tf.placeholder_with_default(False, shape=[])
        self.wide_embed, self.deep_embed = [], []

        # For softmax loss, labels are class indices; otherwise float ratings
        if self.loss_type == "softmax":
            self.labels = tf.placeholder(tf.int32, shape=[None])
        else:
            self.labels = tf.placeholder(tf.float32, shape=[None])

        self._build_user_item()
        if self.sparse:
            self._build_sparse()
        if self.dense:
            self._build_dense()
        # Build user rating vector if needed (for direct use or for stats computation)
        if self.use_user_rating_vector or self.use_user_rating_stats:
            self._build_user_rating_vector()
        if self.use_user_rating_stats:  # True, 'both', 'wide', or 'deep'
            self._build_user_rating_stats()

        wide_embed = tf.concat(self.wide_embed, axis=1)
        deep_embed = tf.concat(self.deep_embed, axis=1)
        deep_layer = dense_nn(
            deep_embed,
            self.hidden_units,
            use_bn=self.use_bn,
            dropout_rate=self.dropout_rate,
            is_training=self.is_training,
            name="deep",
        )

        if self.loss_type == "softmax":
            # For softmax loss: output layer has n_rating_classes neurons
            wide_term = tf_dense(units=self.n_rating_classes, name="wide_term")(
                wide_embed
            )
            deep_term = tf_dense(units=self.n_rating_classes, name="deep_term")(
                deep_layer
            )
            # Logits for softmax
            self.logits = tf.add(wide_term, deep_term, name="logits")
            # Probabilities for each rating class
            self.proba_output = tf.nn.softmax(self.logits, name="proba_output")
            # Rating labels as tensor for weighted average computation
            self.rating_labels_tf = tf.constant(
                self.rating_labels, dtype=tf.float32, name="rating_labels"
            )
            # Predicted rating = weighted average of labels by probabilities
            self.output = tf.reduce_sum(
                self.proba_output * self.rating_labels_tf, axis=1, name="output"
            )
        else:
            wide_term = tf_dense(units=1, name="wide_term")(wide_embed)
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

    def _build_dense(self):
        self.dense_values = tf.placeholder(
            tf.float32, shape=[None, self.dense_field_size]
        )
        wide_dense_embed = compute_dense_feats(
            self.dense_values,
            var_name="dense_wide_var",
            var_shape=[self.dense_field_size],
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
        )
        deep_dense_embed = compute_dense_feats(
            self.dense_values,
            var_name="dense_deep_var",
            var_shape=(self.dense_field_size, self.embed_size),
            initializer=tf.glorot_uniform_initializer(),
            regularizer=self.reg,
            flatten=True,
        )
        self.wide_embed.append(wide_dense_embed)
        self.deep_embed.append(deep_dense_embed)

    def _build_user_rating_vector(self):
        """Build user rating vector feature.
        
        This feature represents user's ratings for all items as a dense vector
        of length n_items. During training, the rating for the current item is 
        masked (set to 0) to prevent data leakage. During inference, the full 
        rating vector is used.
        
        The use_user_rating_vector parameter controls which parts use this feature:
        - True or 'both': wide and deep parts
        - 'wide': only wide part (simpler, less prone to overfitting)
        - 'deep': only deep part
        """
        # Determine which parts to use
        mode = self.use_user_rating_vector
        use_wide = mode in (True, 'both', 'wide')
        use_deep = mode in (True, 'both', 'deep')
        
        # Placeholder for user rating vector [batch_size, n_items]
        self.user_rating_vector = tf.placeholder(
            tf.float32, shape=[None, self.n_items], name="user_rating_vector"
        )
        
        # Normalize the rating vector to unit length (L2 normalization)
        # This helps with:
        # 1. Users with many ratings vs few ratings having similar scale
        # 2. Preventing large rating values from dominating
        # Use epsilon to handle all-zero vectors (new users or fully masked)
        rating_vector_norm = tf.nn.l2_normalize(
            self.user_rating_vector, axis=1, epsilon=1e-12
        )
        
        with tf.variable_scope("embedding"):
            if use_wide:
                # Wide part: weighted sum of all normalized ratings -> scalar per sample
                # Shape: [n_items] -> learns importance weight per item
                wide_rating_var = tf.get_variable(
                    name="rating_vector_wide_var",
                    shape=[self.n_items],
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                # [batch, n_items] * [n_items] -> [batch, n_items] -> sum -> [batch, 1]
                wide_rating_embed = tf.reduce_sum(
                    rating_vector_norm * wide_rating_var, axis=1, keepdims=True
                )
                self.wide_embed.append(wide_rating_embed)
            
            if use_deep:
                # Deep part: single projection matrix to compress to embed_size
                # Shape: [n_items, embed_size] -> projects entire rating vector to embed_size
                deep_rating_var = tf.get_variable(
                    name="rating_vector_deep_var",
                    shape=[self.n_items, self.embed_size],
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                # [batch, n_items] @ [n_items, embed_size] -> [batch, embed_size]
                deep_rating_embed = tf.matmul(rating_vector_norm, deep_rating_var)
                self.deep_embed.append(deep_rating_embed)

    def _build_user_rating_stats(self):
        """Build user rating statistics feature (mean, std).
        
        When use_user_rating_vector is also enabled, computes stats from user_rating_vector.
        When only use_user_rating_stats is enabled, uses a separate small placeholder for
        pre-computed stats (much faster, avoids creating full n_items vectors).
        
        The use_user_rating_stats parameter controls which parts use this feature:
        - True or 'both': wide and deep parts
        - 'wide': only wide part
        - 'deep': only deep part
        """
        
        # Determine which parts to use
        mode = self.use_user_rating_stats
        use_wide = mode in (True, 'both', 'wide')
        use_deep = mode in (True, 'both', 'deep')
        
        if self.use_user_rating_vector:
            # Compute mean and std from user_rating_vector in TensorFlow
            non_zero_mask = tf.not_equal(self.user_rating_vector, 0.0)
            non_zero_count = tf.reduce_sum(tf.cast(non_zero_mask, tf.float32), axis=1, keepdims=True)
            non_zero_count = tf.maximum(non_zero_count, 1.0)
            
            rating_sum = tf.reduce_sum(self.user_rating_vector, axis=1, keepdims=True)
            user_mean = rating_sum / non_zero_count
            
            squared_ratings = tf.square(self.user_rating_vector)
            squared_sum = tf.reduce_sum(squared_ratings, axis=1, keepdims=True)
            mean_of_squares = squared_sum / non_zero_count
            variance = tf.maximum(mean_of_squares - tf.square(user_mean), 0.0)
            user_std = tf.sqrt(variance)
            
            user_rating_stats = tf.concat([user_mean, user_std], axis=1)
        else:
            # Use pre-computed stats from collator (much faster)
            self.user_rating_stats = tf.placeholder(
                tf.float32, shape=[None, 2], name="user_rating_stats"
            )
            user_rating_stats = self.user_rating_stats
        
        with tf.variable_scope("embedding"):
            if use_wide:
                wide_stats_var = tf.get_variable(
                    name="rating_stats_wide_var",
                    shape=[2],
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                wide_stats_embed = tf.reduce_sum(
                    user_rating_stats * wide_stats_var, axis=1, keepdims=True
                )
                self.wide_embed.append(wide_stats_embed)
            
            if use_deep:
                deep_stats_var = tf.get_variable(
                    name="rating_stats_deep_var",
                    shape=[2, self.embed_size],
                    initializer=tf.glorot_uniform_initializer(),
                    regularizer=self.reg,
                )
                deep_stats_embed = tf.matmul(user_rating_stats, deep_stats_var)
                self.deep_embed.append(deep_stats_embed)

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

    def fit(
        self,
        train_data,
        neg_sampling,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        num_workers=0,
    ):
        """Fit Wide & Deep model on the training data.

        For softmax loss, rating labels are automatically extracted from the
        unique values in the training data labels.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Data object used for training.
        neg_sampling : bool
            Whether to perform negative sampling for training or evaluating data.
        verbose : int, default: 1
            Print verbosity.
        shuffle : bool, default: True
            Whether to shuffle the training data.
        eval_data : :class:`~libreco.data.TransformedSet` object, default: None
            Data object used for evaluating.
        metrics : list or None, default: None
            List of metrics for evaluating.
        k : int, default: 10
            Parameter of metrics, e.g. recall at k, ndcg at k
        eval_batch_size : int, default: 8192
            Batch size for evaluating.
        eval_user_num : int or None, default: None
            Number of users for evaluating.
        num_workers : int, default: 0
            How many subprocesses to use for training data loading.
        """
        # Extract rating labels from training data for softmax loss
        if self.loss_type == "softmax" and self.rating_labels is None:
            self._setup_rating_labels(train_data)

        # Store sparse_interaction for user rating vector/stats features
        if (self.use_user_rating_vector or self.use_user_rating_stats) and self.sparse_interaction is None:
            self.sparse_interaction = train_data.sparse_interaction

        # Call parent fit()
        super().fit(
            train_data,
            neg_sampling,
            verbose,
            shuffle,
            eval_data,
            metrics,
            k,
            eval_batch_size,
            eval_user_num,
            num_workers,
        )

    def _setup_rating_labels(self, train_data):
        """Extract unique rating labels from training data for softmax loss.

        Parameters
        ----------
        train_data : :class:`~libreco.data.TransformedSet` object
            Training data containing labels.
        """
        unique_labels = np.unique(train_data.labels)
        self.rating_labels = np.array(sorted(unique_labels), dtype=np.float32)
        self.n_rating_classes = len(self.rating_labels)
        self.rating_label_to_index = {
            label: idx for idx, label in enumerate(self.rating_labels)
        }
        print(
            f"Softmax loss: detected {self.n_rating_classes} rating classes "
            f"from training data: {self.rating_labels.tolist()}"
        )

    def predict_proba(self, user, item, feats=None, cold_start="average", inner_id=False):
        """Get probability distribution over rating classes.

        This method is only available when ``loss_type='softmax'``.

        Parameters
        ----------
        user : int or str or array_like
            User id or batch of user ids.
        item : int or str or array_like
            Item id or batch of item ids.
        feats : dict or None, default: None
            Extra features used in prediction.
        cold_start : {'popular', 'average'}, default: 'average'
            Cold start strategy.
        inner_id : bool, default: False
            Whether to use inner_id defined in `libreco`.

        Returns
        -------
        dict
            Dictionary with:
            - 'labels': numpy array of rating labels
            - 'probabilities': numpy array of shape (n_samples, n_classes) with
              probability for each rating class

        Raises
        ------
        ValueError
            If called on a model not using softmax loss.
        """
        if self.loss_type != "softmax":
            raise ValueError(
                "predict_proba is only available when loss_type='softmax'. "
                f"Current loss_type is '{self.loss_type}'."
            )

        from ..prediction.predict import get_user_rating_vectors_for_inference
        from ..prediction.preprocess import convert_id, get_cached_seqs, get_original_feats, set_temp_feats
        from ..tfops.features import get_feed_dict
        from ..utils.validate import check_unknown

        user, item = convert_id(self, user, item, inner_id)
        unknown_num, unknown_index, user, item = check_unknown(self, user, item)
        has_sparse = self.sparse if hasattr(self, "sparse") else None
        has_dense = self.dense if hasattr(self, "dense") else None
        (
            user_indices,
            item_indices,
            sparse_indices,
            dense_values,
        ) = get_original_feats(self.data_info, user, item, has_sparse, has_dense)

        if feats is not None:
            assert isinstance(feats, dict), "`feats` must be `dict`."
            assert len(user_indices) == 1, "Predict with feats only supports single user."
            sparse_indices, dense_values = set_temp_feats(
                self.data_info, sparse_indices, dense_values, feats
            )

        # Get full user rating vectors for inference (without masking)
        user_rating_vectors = get_user_rating_vectors_for_inference(self, user_indices)

        seqs, seq_len = get_cached_seqs(self, user_indices, repeat=False)
        feed_dict = get_feed_dict(
            model=self,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=seqs,
            user_interacted_len=seq_len,
            user_rating_vectors=user_rating_vectors,
            is_training=False,
        )
        proba = self.sess.run(self.proba_output, feed_dict)

        # Handle unknown users/items by returning uniform probabilities
        if unknown_num > 0 and cold_start == "popular":
            uniform_proba = np.ones(self.n_rating_classes) / self.n_rating_classes
            for i in unknown_index:
                proba[i] = uniform_proba

        return {
            "labels": self.rating_labels.copy(),
            "probabilities": proba,
        }
