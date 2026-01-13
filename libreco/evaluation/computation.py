import numpy as np
import pandas as pd
from tqdm import tqdm

from ..data import TransformedEvalSet
from ..prediction.preprocess import convert_id
from ..utils.validate import check_labels


def build_eval_transformed_data(model, data, neg_sampling, seed):
    if isinstance(data, pd.DataFrame):
        assert "user" in data and "item" in data and "label" in data
        users = data["user"].tolist()
        items = data["item"].tolist()
        user_indices, item_indices = convert_id(model, users, items, inner_id=False)
        labels = data["label"].to_numpy(dtype=np.float32)
        
        # Extract interaction features if model uses them and data has them
        interaction_sparse_indices = None
        interaction_dense_values = None
        if hasattr(model, 'interaction_sparse') and model.interaction_sparse:
            interaction_sparse_col = model.data_info.interaction_sparse_col.name
            if all(col in data.columns for col in interaction_sparse_col):
                # Transform sparse features to indices
                interaction_sparse_indices = _transform_eval_interaction_sparse(
                    data, interaction_sparse_col, model.data_info
                )
        if hasattr(model, 'interaction_dense') and model.interaction_dense:
            interaction_dense_col = model.data_info.interaction_dense_col.name
            if all(col in data.columns for col in interaction_dense_col):
                interaction_dense_values = data[interaction_dense_col].to_numpy(dtype=np.float32)
        
        data = TransformedEvalSet(
            user_indices, item_indices, labels,
            interaction_sparse_indices=interaction_sparse_indices,
            interaction_dense_values=interaction_dense_values,
        )
    if neg_sampling and not data.has_sampled:
        num_neg = model.num_neg or 1 if hasattr(model, "num_neg") else 1
        data.build_negatives(model.n_items, num_neg, seed=seed)
    else:
        check_labels(model, data.labels, neg_sampling)
    return data


def _transform_eval_interaction_sparse(data, col_names, data_info):
    """Transform interaction sparse features to indices for evaluation."""
    indices = []
    for col in col_names:
        unique_vals = data_info.interaction_sparse_unique_vals.get(col, {})
        # Map values to indices, using 0 for unknown values
        col_indices = data[col].map(lambda x: unique_vals.get(x, 0)).to_numpy(dtype=np.int32)
        indices.append(col_indices)
    return np.column_stack(indices) if indices else None


def compute_preds(model, data, batch_size):
    y_pred = list()
    y_label = list()
    y_item_indices = list()
    
    # Check if we have interaction features in eval data
    has_interaction_feats = (
        hasattr(data, 'get_interaction_features') and
        (data.interaction_sparse_indices is not None or 
         data.interaction_dense_values is not None)
    )
    
    for i in tqdm(range(0, len(data), batch_size), desc="eval_pointwise"):
        user_indices, item_indices, labels = data[i : i + batch_size]
        
        if has_interaction_feats:
            interaction_sparse, interaction_dense = data.get_interaction_features(
                slice(i, i + batch_size)
            )
            preds = predict_with_interaction_feats(
                model, user_indices, item_indices, 
                interaction_sparse, interaction_dense
            )
        else:
            preds = model.predict(user_indices, item_indices, inner_id=True)
        
        y_pred.extend(preds)
        y_label.extend(labels)
        y_item_indices.extend(item_indices)
    return y_pred, y_label, y_item_indices


def predict_with_interaction_feats(model, user_indices, item_indices, 
                                   interaction_sparse, interaction_dense):
    """Predict with interaction features for evaluation."""
    from ..prediction.preprocess import get_original_feats
    from ..tfops.features import get_feed_dict
    from ..prediction.predict import (
        get_cached_seqs, get_user_rating_vectors_for_inference,
        get_user_rating_stats_for_inference, normalize_prediction
    )
    
    has_sparse = model.sparse if hasattr(model, "sparse") else None
    has_dense = model.dense if hasattr(model, "dense") else None
    (
        user_idx, item_idx, sparse_indices, dense_values
    ) = get_original_feats(model.data_info, user_indices, item_indices, has_sparse, has_dense)
    
    seqs, seq_len = get_cached_seqs(model, user_idx, repeat=False)
    user_rating_vectors = get_user_rating_vectors_for_inference(model, user_idx)
    user_rating_stats = get_user_rating_stats_for_inference(model, user_idx)
    
    feed_dict = get_feed_dict(
        model=model,
        user_indices=user_idx,
        item_indices=item_idx,
        sparse_indices=sparse_indices,
        dense_values=dense_values,
        user_interacted_seq=seqs,
        user_interacted_len=seq_len,
        user_rating_vectors=user_rating_vectors,
        user_rating_stats=user_rating_stats,
        interaction_sparse_indices=interaction_sparse,
        interaction_dense_values=interaction_dense,
        is_training=False,
    )
    preds = model.sess.run(model.output, feed_dict)
    return normalize_prediction(preds, model, cold_start="average", unknown_num=0, unknown_index=[])


def compute_probs(model, data, batch_size):
    y_pred, y_label, _ = compute_preds(model, data, batch_size)
    return y_pred, y_label


def compute_recommends(model, users, k, num_batch_users):
    y_recommends = dict()
    for i in tqdm(range(0, len(users), num_batch_users), desc="eval_listwise"):
        batch_users = users[i : i + num_batch_users]
        batch_recs = model.recommend_user(
            user=batch_users,
            n_rec=k,
            inner_id=True,
            filter_consumed=True,
            random_rec=False,
        )
        y_recommends.update(batch_recs)
    return y_recommends
