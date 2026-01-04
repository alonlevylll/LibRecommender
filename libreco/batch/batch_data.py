import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler

from .collators import BaseCollator as NormalCollator
from .collators import (
    GraphCollator,
    GraphDGLCollator,
    LazyCollator,
    LazyPairwiseCollator,
    LazyPointwiseCollator,
    PairwiseCollator,
    PointwiseCollator,
    SparseCollator,
)
from .enums import Backend
from ..utils.constants import FeatModels, SageModels, TfTrainModels
from ..utils.validate import is_listwise_training


class BatchData(torch.utils.data.Dataset):
    def __init__(self, data, use_features, factor=None):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.sparse_indices = data.sparse_indices
        self.dense_values = data.dense_values
        self.interaction_sparse_indices = getattr(data, 'interaction_sparse_indices', None)
        self.interaction_dense_values = getattr(data, 'interaction_dense_values', None)
        self.use_features = use_features
        self.factor = factor
        self.is_lazy = False

    def __getitem__(self, idx):
        batch = {
            "user": self.user_indices[idx],
            "item": self.item_indices[idx],
            "label": self.labels[idx],
        }
        if self.use_features and self.sparse_indices is not None:
            batch["sparse"] = self.sparse_indices[idx]
        if self.use_features and self.dense_values is not None:
            batch["dense"] = self.dense_values[idx]
        if self.use_features and self.interaction_sparse_indices is not None:
            batch["interaction_sparse"] = self.interaction_sparse_indices[idx]
        if self.use_features and self.interaction_dense_values is not None:
            batch["interaction_dense"] = self.interaction_dense_values[idx]
        return batch

    def __len__(self):
        length = len(self.labels)
        return math.ceil(length / self.factor) if self.factor is not None else length


class LazyBatchData(torch.utils.data.Dataset):
    """Memory-efficient dataset that provides only user/item indices per batch.

    Features are joined on-the-fly by the collator during batch processing.
    This significantly reduces memory usage for large datasets.

    Parameters
    ----------
    data : LazyTransformedSet
        Lazy transformed data containing user_indices, item_indices, labels.
    use_features : bool
        Whether the model uses features.
    factor : int or None
        Factor for adjusting dataset length (used in some models).
    """

    def __init__(self, data, use_features, factor=None):
        self.user_indices = data.user_indices
        self.item_indices = data.item_indices
        self.labels = data.labels
        self.use_features = use_features
        self.factor = factor
        self.is_lazy = True
        # Store references to feature DataFrames
        self.user_features_df = data.user_features_df
        self.item_features_df = data.item_features_df
        # Store column info
        self.sparse_col = data.sparse_col
        self.dense_col = data.dense_col
        self.multi_sparse_col = data.multi_sparse_col
        self.user_sparse_col = data.user_sparse_col
        self.user_dense_col = data.user_dense_col
        self.item_sparse_col = data.item_sparse_col
        self.item_dense_col = data.item_dense_col

    def __getitem__(self, idx):
        """Return batch of user/item indices and labels without features.

        Features will be joined by the collator.
        """
        batch = {
            "user": self.user_indices[idx],
            "item": self.item_indices[idx],
            "label": self.labels[idx],
            "is_lazy": True,
        }
        return batch

    def __len__(self):
        length = len(self.labels)
        return math.ceil(length / self.factor) if self.factor is not None else length


def get_batch_loader(model, data, neg_sampling, batch_size, shuffle, num_workers, seed):
    torch.manual_seed(seed)
    use_features = True if FeatModels.contains(model.model_name) else False
    factor = (
        model.num_walks * model.sample_walk_len
        if SageModels.contains(model.model_name) and model.paradigm == "i2i"
        else None
    )

    # Check if data is lazy mode
    is_lazy = getattr(data, "is_lazy", False)

    if is_lazy:
        batch_data = LazyBatchData(data, use_features, factor)
        collate_fn = get_lazy_collate_fn(model, data, neg_sampling, num_workers)
    else:
        batch_data = BatchData(data, use_features, factor)
        collate_fn = get_collate_fn(model, neg_sampling, num_workers)

    # Check for positive sampler (weighted sampling for unpopular items)
    pos_sampler = getattr(model, "pos_sampler", "random")
    
    if shuffle:
        if pos_sampler == "unpopular":
            # Compute sample weights: 1/sqrt(item_count) - same as WMSE
            sample_weights = _compute_sample_weights(data.item_indices, model.n_items)
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(batch_data),
                replacement=True,
            )
        else:
            sampler = RandomSampler(batch_data)
    else:
        sampler = SequentialSampler(batch_data)
    
    batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=False)
    return DataLoader(
        batch_data,
        batch_size=None,  # `batch_size=None` disables automatic batching
        sampler=batch_sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def _compute_sample_weights(item_indices, n_items):
    """Compute sample weights inversely proportional to item frequency.
    
    Uses the same weighting as WMSE: weight = 1 / sqrt(count).
    Low-volume items get higher sampling probability.
    
    Parameters
    ----------
    item_indices : array-like
        Item indices for all training samples.
    n_items : int
        Total number of items.
    
    Returns
    -------
    torch.Tensor
        Weight for each training sample.
    """
    # Count frequency of each item
    item_counts = np.bincount(item_indices, minlength=n_items).astype(np.float32)
    
    # Handle zero counts
    item_counts[item_counts == 0] = 1.0
    
    # Compute weight per item: 1 / sqrt(count)
    item_weights = 1.0 / np.sqrt(item_counts)
    
    # Get weight for each sample based on its item
    sample_weights = item_weights[item_indices]
    
    return torch.from_numpy(sample_weights)


def get_collate_fn(model, neg_sampling, num_workers):
    model_name, data_info = model.model_name, model.data_info
    backend = Backend.TF if TfTrainModels.contains(model_name) else Backend.TORCH
    separate_features = True if model_name == "TwoTower" else False
    if model_name == "YouTubeRetrieval":
        collate_fn = SparseCollator(model, data_info, backend)
    elif model_name == "TwoTower" and model.loss_type == "softmax":
        collate_fn = NormalCollator(model, data_info, backend, separate_features)
    elif SageModels.contains(model_name):
        if model.use_dgl:
            assert num_workers == 0, "DGL models can't use multiprocessing data loader"
            collate_fn = GraphDGLCollator(model, data_info, backend)
        else:
            collate_fn = GraphCollator(model, data_info, backend)
    elif model.task == "rating" or not neg_sampling:
        collate_fn = NormalCollator(model, data_info, backend, separate_features)
    else:
        if model.loss_type in ("cross_entropy", "focal"):
            collate_fn = PointwiseCollator(model, data_info, backend, separate_features)
        else:
            repeat_positives = True if backend is Backend.TF else False
            collate_fn = PairwiseCollator(model, data_info, backend, repeat_positives)
    return collate_fn


def get_lazy_collate_fn(model, data, neg_sampling, num_workers):
    """Get collate function for lazy loading mode.

    In lazy mode, features are joined on-the-fly during batch processing.
    """
    model_name, data_info = model.model_name, model.data_info
    backend = Backend.TF if TfTrainModels.contains(model_name) else Backend.TORCH
    separate_features = True if model_name == "TwoTower" else False

    # For lazy loading, use lazy collators that perform feature joins
    if model_name == "YouTubeRetrieval":
        # YouTubeRetrieval uses SparseCollator - not supported in lazy mode yet
        raise NotImplementedError(
            "YouTubeRetrieval model is not yet supported in lazy loading mode."
        )
    elif SageModels.contains(model_name):
        # Graph models not supported in lazy mode yet
        raise NotImplementedError(
            "GraphSage models are not yet supported in lazy loading mode."
        )
    elif model.task == "rating" or not neg_sampling:
        collate_fn = LazyCollator(model, data_info, data, backend, separate_features)
    else:
        if model.loss_type in ("cross_entropy", "focal"):
            collate_fn = LazyPointwiseCollator(
                model, data_info, data, backend, separate_features
            )
        else:
            repeat_positives = True if backend is Backend.TF else False
            collate_fn = LazyPairwiseCollator(
                model, data_info, data, backend, repeat_positives
            )
    return collate_fn


# consider negative sampling and random walks in batch_size
def adjust_batch_size(model, original_batch_size):
    if is_listwise_training(model):
        return original_batch_size
    elif SageModels.contains(model.model_name) and model.paradigm == "i2i":
        walk_len = model.sample_walk_len
        bs = original_batch_size / model.num_neg / model.num_walks / walk_len
        return max(1, int(bs))
    elif model.sampler is not None:
        if model.loss_type in ("cross_entropy", "focal"):
            return max(1, int(original_batch_size / (model.num_neg + 1)))
        else:
            return max(1, int(original_batch_size / model.num_neg))
    return original_batch_size
