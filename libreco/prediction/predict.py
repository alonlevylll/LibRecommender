import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm import tqdm

from .preprocess import (
    convert_id,
    features_from_batch,
    get_cached_dual_seq,
    get_cached_seqs,
    get_original_feats,
    set_temp_feats,
)
from ..tfops.features import get_dual_seq_feed_dict, get_feed_dict
from ..utils.validate import check_unknown


def get_user_rating_vectors_for_inference(model, user_indices):
    """Get full user rating vectors for inference (without masking).
    
    Parameters
    ----------
    model : object
        Model object with sparse_interaction attribute.
    user_indices : array_like
        Array of user indices.
        
    Returns
    -------
    numpy.ndarray or None
        Array of shape [batch_size, n_items] with full user rating vectors,
        or None if neither use_user_rating_vector nor use_user_rating_stats is enabled.
    """
    use_rating_vector = getattr(model, "use_user_rating_vector", False)
    use_rating_stats = getattr(model, "use_user_rating_stats", False)
    # Rating vectors are needed for both direct use and for computing stats
    if not (use_rating_vector or use_rating_stats):
        return None
    if model.sparse_interaction is None:
        return None
    
    user_indices = np.asarray(user_indices)
    batch_size = len(user_indices)
    n_items = model.n_items
    
    user_rating_vectors = np.zeros((batch_size, n_items), dtype=np.float32)
    
    # The sparse matrix might have different shape than n_items
    sparse_n_rows = model.sparse_interaction.shape[0]
    copy_cols = min(model.sparse_interaction.shape[1], n_items)
    
    for i, user_idx in enumerate(user_indices):
        if user_idx < sparse_n_rows:
            # Get the user's row from sparse matrix as dense array
            user_row = model.sparse_interaction[user_idx].toarray().flatten()
            # Copy only the valid range (handle shape mismatch)
            user_rating_vectors[i, :copy_cols] = user_row[:copy_cols]
    
    return user_rating_vectors


def normalize_prediction(preds, model, cold_start, unknown_num, unknown_index):
    if model.task == "rating":
        preds = np.clip(preds, model.lower_bound, model.upper_bound)
    elif model.task == "ranking":
        # preds = 1 / (1 + np.exp(-z))
        preds = expit(preds)

    if unknown_num > 0 and cold_start == "popular":
        if isinstance(preds, np.ndarray):
            preds[unknown_index] = model.default_pred
        elif isinstance(preds, list):
            for i in unknown_index:
                preds[i] = model.default_pred
        else:
            preds = model.default_pred
    return preds


def predict_from_embedding(model, user, item, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    preds = np.sum(model.user_embeds_np[user] * model.item_embeds_np[item], axis=1)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_tf_feat(model, user, item, feats, cold_start, inner_id):
    user, item = convert_id(model, user, item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    has_sparse = model.sparse if hasattr(model, "sparse") else None
    has_dense = model.dense if hasattr(model, "dense") else None
    (
        user_indices,
        item_indices,
        sparse_indices,
        dense_values,
    ) = get_original_feats(model.data_info, user, item, has_sparse, has_dense)

    if feats is not None:
        assert isinstance(feats, dict), "`feats` must be `dict`."
        assert len(user_indices) == 1, "Predict with feats only supports single user."
        sparse_indices, dense_values = set_temp_feats(
            model.data_info, sparse_indices, dense_values, feats
        )

    # Get full user rating vectors for inference (without masking)
    # Note: user_rating_stats are computed from this in the TF graph
    user_rating_vectors = get_user_rating_vectors_for_inference(model, user_indices)

    if model.model_name == "SIM":
        long_seqs, long_lens, short_seqs, short_lens = get_cached_dual_seq(
            model, user_indices, repeat=False
        )
        feed_dict = get_dual_seq_feed_dict(
            model,
            user_indices,
            item_indices,
            sparse_indices,
            dense_values,
            long_seqs,
            long_lens,
            short_seqs,
            short_lens,
            is_training=False,
        )
        preds = model.sess.run(model.inference_output, feed_dict)
    else:
        seqs, seq_len = get_cached_seqs(model, user_indices, repeat=False)
        feed_dict = get_feed_dict(
            model=model,
            user_indices=user_indices,
            item_indices=item_indices,
            sparse_indices=sparse_indices,
            dense_values=dense_values,
            user_interacted_seq=seqs,
            user_interacted_len=seq_len,
            user_rating_vectors=user_rating_vectors,
            is_training=False,
        )
        preds = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)


def predict_data_with_feats(
    model, data, batch_size=None, cold_start="average", inner_id=False
):
    assert isinstance(data, pd.DataFrame), "Data must be pandas DataFrame"
    user, item = convert_id(model, data.user, data.item, inner_id)
    unknown_num, unknown_index, user, item = check_unknown(model, user, item)
    if not batch_size:
        batch_size = len(data)
    preds = np.zeros(len(data), dtype=np.float32)
    for i in tqdm(range(0, len(data), batch_size), "pred_data"):
        batch_slice = slice(i, i + batch_size)
        batch_data = data.iloc[batch_slice]
        user_indices = user[batch_slice]
        item_indices = item[batch_slice]
        sparse_indices, dense_values = features_from_batch(
            model.data_info, model.sparse, model.dense, batch_data
        )
        # Get full user rating vectors for inference (without masking)
        user_rating_vectors = get_user_rating_vectors_for_inference(model, user_indices)

        if model.model_name == "SIM":
            long_seqs, long_lens, short_seqs, short_lens = get_cached_dual_seq(
                model, user_indices, repeat=False
            )
            feed_dict = get_dual_seq_feed_dict(
                model,
                user_indices,
                item_indices,
                sparse_indices,
                dense_values,
                long_seqs,
                long_lens,
                short_seqs,
                short_lens,
                is_training=False,
            )
            preds[batch_slice] = model.sess.run(model.inference_output, feed_dict)
        else:
            seqs, seq_len = get_cached_seqs(model, user_indices, repeat=False)
            feed_dict = get_feed_dict(
                model=model,
                user_indices=user_indices,
                item_indices=item_indices,
                sparse_indices=sparse_indices,
                dense_values=dense_values,
                user_interacted_seq=seqs,
                user_interacted_len=seq_len,
                user_rating_vectors=user_rating_vectors,
                is_training=False,
            )
            preds[batch_slice] = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start, unknown_num, unknown_index)
