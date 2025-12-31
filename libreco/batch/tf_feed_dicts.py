from .batch_unit import (
    PairwiseBatch,
    PointwiseBatch,
    PointwiseDualSeqBatch,
    PointwiseSepFeatBatch,
    SparseBatch,
)
from ..feature.ssl import get_ssl_features
from ..utils.constants import SequenceModels


def get_tf_feeds(model, data, is_training):
    if isinstance(data, SparseBatch):
        return _sparse_feed_dict(model, data, is_training)
    elif isinstance(data, PointwiseDualSeqBatch):
        return _dual_seq_feed_dict(model, data, is_training)
    elif isinstance(data, PointwiseSepFeatBatch):
        return _separate_feed_dict(model, data, is_training)
    elif isinstance(data, PairwiseBatch):
        return _pairwise_feed_dict(model, data, is_training)
    else:
        return _pointwise_feed_dict(model, data, is_training)


def _sparse_feed_dict(model, data: SparseBatch, is_training):
    feed_dict = {
        model.item_interaction_indices: data.seqs.interacted_indices,
        model.item_interaction_values: data.seqs.interacted_values,
        model.modified_batch_size: data.seqs.modified_batch_size,
        model.item_indices: data.items,
        model.is_training: is_training,
    }
    if hasattr(model, "user_sparse") and model.user_sparse:
        feed_dict.update({model.user_sparse_indices: data.sparse_indices})
    if hasattr(model, "user_dense") and model.user_dense:
        feed_dict.update({model.user_dense_values: data.dense_values})
    return feed_dict


def _pairwise_feed_dict(model, data: PairwiseBatch, is_training):
    if model.model_name == "BPR":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    elif model.model_name == "RNN4Rec":
        feed_dict = {
            model.user_interacted_seq: data.seqs.interacted_seq,
            model.user_interacted_len: data.seqs.interacted_len,
            model.item_indices_pos: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
    elif model.model_name == "TwoTower":
        feed_dict = {
            model.user_indices: data.queries,
            model.item_indices: data.item_pairs[0],
            model.item_indices_neg: data.item_pairs[1],
        }
        if model.user_sparse:
            feed_dict.update(
                {model.user_sparse_indices: data.sparse_indices.query_feats}
            )
        if model.user_dense:
            feed_dict.update({model.user_dense_values: data.dense_values.query_feats})
        if model.item_sparse:
            feed_dict.update(
                {model.item_sparse_indices: data.sparse_indices.item_pos_feats}
            )
            feed_dict.update(
                {model.item_sparse_indices_neg: data.sparse_indices.item_neg_feats}
            )
        if model.item_dense:
            feed_dict.update(
                {model.item_dense_values: data.dense_values.item_pos_feats}
            )
            feed_dict.update(
                {model.item_dense_values_neg: data.dense_values.item_neg_feats}
            )
    else:
        raise ValueError(
            "Only `BPR`, `RNN4Rec` and `TwoTower` use pairwise loss in tf models"
        )
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    return feed_dict


def _convert_labels_to_indices(model, labels):
    """Convert rating labels to class indices for softmax loss.
    
    Parameters
    ----------
    model : object
        Model with rating_label_to_index mapping.
    labels : array_like
        Rating labels to convert.
    
    Returns
    -------
    numpy.ndarray
        Integer class indices.
    """
    import numpy as np
    labels_arr = np.asarray(labels)
    # Vectorized: find the closest rating class for each label
    # rating_labels shape: (n_classes,), labels_arr shape: (batch_size,)
    # Compute absolute difference and find argmin
    diff = np.abs(labels_arr[:, np.newaxis] - model.rating_labels[np.newaxis, :])
    indices = np.argmin(diff, axis=1).astype(np.int32)
    return indices


def _is_softmax_rating_model(model):
    """Check if model uses softmax loss for rating prediction.
    
    This is a strict check to ensure we only convert labels for models
    that are explicitly configured for softmax rating classification.
    """
    # Must have loss_type attribute set to "softmax"
    if not hasattr(model, "loss_type") or model.loss_type != "softmax":
        return False
    # Must have rating task
    if not hasattr(model, "task") or model.task != "rating":
        return False
    # Must have rating_label_to_index mapping set up
    if not hasattr(model, "rating_label_to_index") or model.rating_label_to_index is None:
        return False
    # Must have rating_labels array
    if not hasattr(model, "rating_labels") or model.rating_labels is None:
        return False
    return True


def _pointwise_feed_dict(model, data: PointwiseBatch, is_training):
    feed_dict = dict()
    if hasattr(model, "user_indices"):
        feed_dict.update({model.user_indices: data.users})
    if hasattr(model, "item_indices"):
        feed_dict.update({model.item_indices: data.items})
    if hasattr(model, "labels"):
        # For softmax loss in rating task, convert labels to class indices
        if _is_softmax_rating_model(model):
            label_indices = _convert_labels_to_indices(model, data.labels)
            feed_dict.update({model.labels: label_indices})
        else:
            feed_dict.update({model.labels: data.labels})
    if hasattr(model, "is_training"):
        feed_dict.update({model.is_training: is_training})
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    if SequenceModels.contains(model.model_name):
        feed_dict.update(
            {
                model.user_interacted_seq: data.seqs.interacted_seq,
                model.user_interacted_len: data.seqs.interacted_len,
            }
        )
    # User rating vector / stats features
    use_rating_vector = getattr(model, "use_user_rating_vector", False)
    use_rating_stats = getattr(model, "use_user_rating_stats", False)
    
    # When use_user_rating_vector is enabled, pass full vectors
    if use_rating_vector and hasattr(data, "user_rating_vectors") and data.user_rating_vectors is not None:
        feed_dict.update({model.user_rating_vector: data.user_rating_vectors})
    # When only use_user_rating_stats is enabled (without vectors), pass pre-computed stats
    elif use_rating_stats and not use_rating_vector and hasattr(data, "user_rating_stats") and data.user_rating_stats is not None:
        feed_dict.update({model.user_rating_stats: data.user_rating_stats})
    
    return feed_dict


def _separate_feed_dict(model, data: PointwiseSepFeatBatch, is_training):
    feed_dict = {
        model.user_indices: data.users,
        model.item_indices: data.items,
        model.is_training: is_training,
    }
    if hasattr(model, "labels"):
        feed_dict.update({model.labels: data.labels})
    if hasattr(model, "correction"):
        feed_dict.update({model.correction: model.item_corrections[data.items]})
    if model.user_sparse:
        feed_dict.update({model.user_sparse_indices: data.sparse_indices.user_feats})
    if model.user_dense:
        feed_dict.update({model.user_dense_values: data.dense_values.user_feats})
    if model.item_sparse:
        feed_dict.update({model.item_sparse_indices: data.sparse_indices.item_feats})
    if model.item_dense:
        feed_dict.update({model.item_dense_values: data.dense_values.item_feats})
    if hasattr(model, "ssl_pattern") and model.ssl_pattern is not None:
        ssl_feats = get_ssl_features(model, len(data.items))
        feed_dict.update(ssl_feats)
    return feed_dict


def _dual_seq_feed_dict(model, data: PointwiseDualSeqBatch, is_training):
    feed_dict = {
        model.user_indices: data.users,
        model.item_indices: data.items,
        model.labels: data.labels,
        model.is_training: is_training,
        model.long_seqs: data.seqs.long_seq,
        model.long_seq_lens: data.seqs.long_len,
        model.short_seqs: data.seqs.short_seq,
        model.short_seq_lens: data.seqs.short_len,
    }
    if hasattr(model, "sparse") and model.sparse:
        feed_dict.update({model.sparse_indices: data.sparse_indices})
    if hasattr(model, "dense") and model.dense:
        feed_dict.update({model.dense_values: data.dense_values})
    return feed_dict
