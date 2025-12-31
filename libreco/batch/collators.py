import random

import numpy as np
import pandas as pd
import torch

from .batch_unit import (
    DualSeqFeats,
    PairFeats,
    PairwiseBatch,
    PointwiseBatch,
    PointwiseDualSeqBatch,
    PointwiseSepFeatBatch,
    SeqFeats,
    SparseBatch,
    SparseSeqFeats,
    TripleFeats,
)
from .enums import FeatType
from .sequence import get_dual_seqs, get_interacted_seqs, get_sparse_interacted
from ..graph import build_subgraphs, pairs_from_dgl_graph
from ..sampling import (
    neg_probs_from_frequency,
    negatives_from_out_batch,
    negatives_from_popular,
    negatives_from_random,
    negatives_from_unconsumed,
    pairs_from_random_walk,
    pos_probs_from_frequency,
)
from ..utils.constants import SequenceModels


class BaseCollator:
    def __init__(
        self,
        model,
        data_info,
        backend,
        separate_features=False,
        temperature=0.75,
    ):
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_consumed = data_info.user_consumed
        self.item_consumed = data_info.item_consumed
        self.user_sparse_col_index = data_info.user_sparse_col.index
        self.item_sparse_col_index = data_info.item_sparse_col.index
        self.user_dense_col_index = data_info.user_dense_col.index
        self.item_dense_col_index = data_info.item_dense_col.index
        self.item_sparse_unique = data_info.item_sparse_unique
        self.item_dense_unique = data_info.item_dense_unique
        self.has_seq = True if SequenceModels.contains(model.model_name) else False
        self.seq_mode = model.seq_mode if hasattr(model, "seq_mode") else None
        self.max_seq_len = model.max_seq_len if hasattr(model, "max_seq_len") else None
        self.dual_seq = True if model.model_name == "SIM" else False
        self.long_max_len = model.long_max_len if self.dual_seq else None
        self.short_max_len = model.short_max_len if self.dual_seq else None
        self.separate_features = separate_features
        self.backend = backend
        self.seed = model.seed
        self.temperature = temperature
        self.user_consumed_set = None
        self.neg_probs = None
        self.np_rng = None

    def __call__(self, batch):
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch
        batch_data = batch_cls(
            users=batch["user"],
            items=batch["item"],
            labels=batch["label"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_col_index(self, feat_type):
        if feat_type is FeatType.SPARSE:
            user_col_index = self.user_sparse_col_index
            item_col_index = self.item_sparse_col_index
        elif feat_type is FeatType.DENSE:
            user_col_index = self.user_dense_col_index
            item_col_index = self.item_dense_col_index
        else:
            raise ValueError("`feat_type` must be sparse or dense.")
        return user_col_index, item_col_index

    def get_features(self, batch, feat_type):
        if feat_type.value not in batch:
            return
        features = batch[feat_type.value]
        if self.separate_features:
            user_col_index, item_col_index = self.get_col_index(feat_type)
            user_features = features[:, user_col_index] if user_col_index else None
            item_features = features[:, item_col_index] if item_col_index else None
            features = PairFeats(user_features, item_features)
        return features

    def get_seqs(self, user_indices, item_indices):
        if not self.has_seq:
            return
        self._set_random_seeds()
        self._set_user_consumed()
        if self.dual_seq:
            long_seqs, long_lens, short_seqs, short_lens = get_dual_seqs(
                user_indices,
                item_indices,
                self.user_consumed,
                self.n_items,
                self.long_max_len,
                self.short_max_len,
                self.user_consumed_set,
            )
            return DualSeqFeats(long_seqs, long_lens, short_seqs, short_lens)
        else:
            seqs, seq_lens = get_interacted_seqs(
                user_indices,
                item_indices,
                self.user_consumed,
                self.n_items,
                self.seq_mode,
                self.max_seq_len,
                self.user_consumed_set,
                self.np_rng,
            )
            return SeqFeats(seqs, seq_lens)

    def sample_neg_items(self, batch, sampler, num_neg):
        if sampler == "unconsumed":
            self._set_user_consumed()
            items_neg = negatives_from_unconsumed(
                self.user_consumed_set,
                batch["user"],
                batch["item"],
                self.n_items,
                num_neg,
            )
        elif sampler == "popular":
            self._set_random_seeds()
            self._set_neg_probs()
            items_neg = negatives_from_popular(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
                probs=self.neg_probs,
            )
        else:
            self._set_random_seeds()
            items_neg = negatives_from_random(
                self.np_rng,
                self.n_items,
                batch["item"],
                num_neg,
            )
        return items_neg

    def _set_user_consumed(self):
        if self.user_consumed_set is None:
            self.user_consumed_set = [
                set(self.user_consumed[u]) for u in range(self.n_users)
            ]

    def _set_neg_probs(self):
        if self.neg_probs is None:
            self.neg_probs = neg_probs_from_frequency(
                self.item_consumed, self.n_items, self.temperature
            )

    def _set_random_seeds(self):
        if self.np_rng is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed if worker_info is None else worker_info.seed
            seed = seed % 3407 * 11
            random.seed(seed)
            torch.manual_seed(seed)
            self.np_rng = np.random.default_rng(seed)


class SparseCollator(BaseCollator):
    def __init__(self, model, data_info, backend):
        super().__init__(model, data_info, backend)

    def __call__(self, batch):
        seq_batch = self.get_seqs(batch["user"], batch["item"])
        sparse_batch = self.get_features(batch, FeatType.SPARSE)
        dense_batch = self.get_features(batch, FeatType.DENSE)
        return SparseBatch(
            seqs=seq_batch,
            items=batch["item"],
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
        )

    def get_seqs(self, user_indices, item_indices):
        if self.seq_mode == "random":
            self._set_random_seeds()
        batch_indices, batch_values, batch_size = get_sparse_interacted(
            user_indices,
            item_indices,
            self.user_consumed,
            self.seq_mode,
            self.max_seq_len,
            self.np_rng,
        )
        return SparseSeqFeats(batch_indices, batch_values, batch_size)


class PointwiseCollator(BaseCollator):
    def __init__(self, model, data_info, backend, separate_features=False):
        super().__init__(model, data_info, backend, separate_features)
        self.sampler = model.sampler
        self.num_neg = model.num_neg

    def __call__(self, batch):
        user_batch = np.repeat(batch["user"], self.num_neg + 1)
        item_batch = np.repeat(batch["item"], self.num_neg + 1)
        label_batch = np.zeros_like(item_batch, dtype=np.float32)
        label_batch[:: (self.num_neg + 1)] = 1.0
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
        for i in range(self.num_neg):
            item_batch[(i + 1) :: (self.num_neg + 1)] = items_neg[i :: self.num_neg]

        sparse_batch = self.get_pointwise_feats(batch, FeatType.SPARSE, item_batch)
        dense_batch = self.get_pointwise_feats(batch, FeatType.DENSE, item_batch)
        seq_batch = self.get_seqs(user_batch, item_batch)
        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch
        batch_data = batch_cls(
            users=user_batch,
            items=item_batch,
            labels=label_batch,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_pointwise_feats(self, batch, feat_type, items):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        user_features = repeat_feats(batch_feats, user_col_index, self.num_neg)
        item_features = get_sampled_item_feats(self, item_col_index, items, feat_type)
        if self.separate_features:
            return PairFeats(user_features, item_features)
        if user_col_index and item_col_index:
            return merge_columns(
                user_features, item_features, user_col_index, item_col_index
            )
        return user_features if user_col_index else item_features


class PairwiseCollator(BaseCollator):
    def __init__(self, model, data_info, backend, repeat_positives):
        super().__init__(model, data_info, backend, separate_features=True)
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.repeat_positives = repeat_positives

    def __call__(self, batch):
        if self.repeat_positives and self.num_neg > 1:
            users = np.repeat(batch["user"], self.num_neg)
            items_pos = np.repeat(batch["item"], self.num_neg)
        else:
            users = batch["user"]
            items_pos = batch["item"]
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)

        sparse_batch = self.get_pairwise_feats(batch, FeatType.SPARSE, items_neg)
        dense_batch = self.get_pairwise_feats(batch, FeatType.DENSE, items_neg)
        seq_batch = self.get_seqs(users, items_pos)
        if self.has_seq and not self.repeat_positives and self.num_neg > 1:
            seq_batch = seq_batch.repeat(self.num_neg)
        batch_data = PairwiseBatch(
            queries=users,
            item_pairs=(items_pos, items_neg),
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )
        return batch_data

    def get_pairwise_feats(self, batch, feat_type, items_neg):
        if feat_type.value not in batch:
            return
        batch_feats = batch[feat_type.value]
        user_col_index, item_col_index = self.get_col_index(feat_type)
        if self.repeat_positives and self.num_neg > 1:
            user_feats = repeat_feats(
                batch_feats, user_col_index, self.num_neg, is_pairwise=True
            )
            item_pos_feats = repeat_feats(
                batch_feats, item_col_index, self.num_neg, is_pairwise=True
            )
        else:
            user_feats = batch_feats[:, user_col_index] if user_col_index else None
            item_pos_feats = batch_feats[:, item_col_index] if item_col_index else None
        item_neg_feats = get_sampled_item_feats(
            self, item_col_index, items_neg, feat_type
        )
        return TripleFeats(user_feats, item_pos_feats, item_neg_feats)


class GraphCollator(BaseCollator):
    def __init__(self, model, data_info, backend, alpha=1e-3):
        super().__init__(model, data_info, backend)
        self.neighbor_walker = model.neighbor_walker
        self.paradigm = model.paradigm
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.num_walks = model.num_walks
        self.walk_length = model.sample_walk_len
        self.start_node = model.start_node
        self.focus_start = model.focus_start
        if self.start_node == "unpopular":
            self.pos_probs = pos_probs_from_frequency(
                self.item_consumed, self.n_users, self.n_items, alpha
            )

    def __call__(self, batch):
        self._set_random_seeds()
        if self.paradigm == "u2i":
            users, items_pos = batch["user"], batch["item"]
            items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
            user_data = self.neighbor_walker.get_user_feats(users)
            item_pos_data = self.neighbor_walker(items_pos)
            item_neg_data = self.neighbor_walker(items_neg)
            return user_data, item_pos_data, item_neg_data
        else:
            start_nodes = self.get_start_nodes(batch)
            items, items_pos = pairs_from_random_walk(
                start_nodes,
                self.user_consumed,
                self.item_consumed,
                self.num_walks,
                self.walk_length,
                self.focus_start,
            )
            items_neg = self.sample_i2i_negatives(items, items_pos)
            item_data = self.neighbor_walker(items, items_pos)
            item_pos_data = self.neighbor_walker(items_pos)
            item_neg_data = self.neighbor_walker(items_neg)
            return item_data, item_pos_data, item_neg_data

    # exclude both items and items_pos
    def sample_i2i_negatives(self, items, items_pos):
        if self.sampler == "out-batch":
            items_neg = negatives_from_out_batch(
                self.np_rng, self.n_items, items_pos, items, self.num_neg
            )
        elif self.sampler == "popular":
            items_neg = negatives_from_popular(
                self.np_rng,
                self.n_items,
                items_pos,
                self.num_neg,
                items=items,
                probs=self.neg_probs,
            )
        else:
            items_neg = negatives_from_random(
                self.np_rng,
                self.n_items,
                items_pos,
                self.num_neg,
                items=items,
            )
        return items_neg

    def get_start_nodes(self, batch):
        size = len(batch["item"])
        if self.start_node == "unpopular":
            population = range(self.n_items)
            start_nodes = random.choices(population, weights=self.pos_probs, k=size)
        else:
            start_nodes = self.np_rng.integers(0, self.n_items, size=size)
            start_nodes = start_nodes.tolist()
        return start_nodes


class GraphDGLCollator(GraphCollator):
    def __init__(self, model, data_info, backend, alpha=1e-3):
        super().__init__(model, data_info, backend, alpha)
        self.graph = model.hetero_g
        self.dgl = model._dgl
        self.dgl_seed = None
        if self.start_node == "unpopular":
            self.pos_probs = torch.tensor(self.pos_probs, dtype=torch.float)

    def __call__(self, batch):
        self._set_random_seeds()
        self._set_dgl_seeds()
        if self.paradigm == "u2i":
            users, items_pos = batch["user"], batch["item"]
            items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
            # nodes in pos_graph and neg_graph are same, difference is the connected edges
            pos_graph, neg_graph, *_ = build_subgraphs(
                users, (items_pos, items_neg), self.paradigm, self.num_neg
            )
            # user -> item heterogeneous graph, users on srcdata, items on dstdata
            all_users = pos_graph.srcdata[self.dgl.NID]
            all_items = pos_graph.dstdata[self.dgl.NID]
            user_data = self.neighbor_walker.get_user_feats(all_users)
            item_data = self.neighbor_walker(all_items)
            return user_data, item_data, pos_graph, neg_graph
        else:
            start_nodes = self.get_start_nodes(batch)
            items, items_pos = pairs_from_dgl_graph(
                self.graph,
                start_nodes,
                self.num_walks,
                self.walk_length,
                self.focus_start,
            )
            items_neg = self.sample_i2i_negatives(items, items_pos)
            # nodes in pos_graph and neg_graph are same, difference is the connected edges
            pos_graph, neg_graph, *target_nodes = build_subgraphs(
                items, (items_pos, items_neg), self.paradigm, self.num_neg
            )
            # item -> item homogeneous graph, items on all nodes
            all_items = pos_graph.ndata[self.dgl.NID]
            item_data = self.neighbor_walker(all_items, target_nodes)
            return item_data, pos_graph, neg_graph

    def get_start_nodes(self, batch):
        size = len(batch["item"])
        if self.start_node == "unpopular":
            start_nodes = torch.multinomial(self.pos_probs, size, replacement=True)
        else:
            start_nodes = torch.randint(0, self.n_items, (size,))
        return start_nodes

    def _set_dgl_seeds(self):
        if self.dgl_seed is None:
            worker_info = torch.utils.data.get_worker_info()
            seed = self.seed if worker_info is None else worker_info.seed
            seed = seed % 3407 * 11
            self.dgl.seed(seed)
            self.dgl_seed = True


def repeat_feats(batch_feats, col_index, num_neg, is_pairwise=False):
    if not col_index:
        return
    column_features = batch_feats[:, col_index]
    repeats = num_neg if is_pairwise else num_neg + 1
    return np.repeat(column_features, repeats, axis=0)


def get_sampled_item_feats(collator, item_col_index, items_sampled, feat_type):
    if not item_col_index:
        return
    if feat_type is FeatType.SPARSE:
        item_unique_features = collator.item_sparse_unique
    elif feat_type is FeatType.DENSE:
        item_unique_features = collator.item_dense_unique
    else:
        raise ValueError("`feat_type` must be sparse or dense.")
    return item_unique_features[items_sampled]


def merge_columns(user_features, item_features, user_col_index, item_col_index):
    if len(user_features) != len(item_features):
        raise ValueError(
            f"length of user_features and length of item_features don't match, "
            f"got {len(user_features)} and {len(item_features)}"
        )
    # keep column names in original order
    orig_cols = user_col_index + item_col_index
    col_reindex = np.arange(len(orig_cols))[np.argsort(orig_cols)]
    concat_features = np.concatenate([user_features, item_features], axis=1)
    return concat_features[:, col_reindex]


class LazyFeatureJoiner:
    """High-performance feature joiner using pre-computed numpy index tables.

    This class pre-computes lookup tables indexed by inner user/item IDs,
    enabling O(1) numpy array indexing instead of slow pandas DataFrame lookups.
    
    Memory usage is still efficient because features are stored once per unique
    user/item rather than duplicated per interaction.
    """

    def __init__(
        self,
        data_info,
        train_data,
        user_features_df,
        item_features_df,
    ):
        self.data_info = data_info
        self.n_users = data_info.n_users
        self.n_items = data_info.n_items
        self.user_unique_vals = data_info.user_unique_vals
        self.item_unique_vals = data_info.item_unique_vals
        self.sparse_unique_vals = data_info.sparse_unique_vals
        self.multi_sparse_unique_vals = data_info.multi_sparse_unique_vals
        self.sparse_offset = data_info.sparse_offset

        # Store column info from train data
        self.sparse_col = train_data.sparse_col
        self.dense_col = train_data.dense_col
        self.multi_sparse_col = train_data.multi_sparse_col
        self.user_sparse_col = train_data.user_sparse_col
        self.user_dense_col = train_data.user_dense_col
        self.item_sparse_col = train_data.item_sparse_col
        self.item_dense_col = train_data.item_dense_col

        # Build sparse index mappings (value -> index) for fast lookup
        self._sparse_idx_mapping = {}
        self._sparse_oov = {}
        if self.sparse_unique_vals:
            for col, vals in self.sparse_unique_vals.items():
                self._sparse_idx_mapping[col] = dict(zip(vals, range(len(vals))))
                self._sparse_oov[col] = len(vals)

        if self.multi_sparse_unique_vals:
            for col, vals in self.multi_sparse_unique_vals.items():
                self._sparse_idx_mapping[col] = dict(zip(vals, range(len(vals))))
                self._sparse_oov[col] = len(vals)

        # Build column to offset mapping
        self._col_offset = self._build_col_offset()

        # Pre-compute numpy lookup tables indexed by inner IDs
        # These enable O(1) lookup: features[inner_id] instead of DataFrame.loc
        self._user_sparse_table = None  # [n_users, num_user_sparse_cols]
        self._user_dense_table = None   # [n_users, num_user_dense_cols]
        self._item_sparse_table = None  # [n_items, num_item_sparse_cols]
        self._item_dense_table = None   # [n_items, num_item_dense_cols]

        # Build the lookup tables
        self._build_lookup_tables(user_features_df, item_features_df)

        # Pre-compute column merge indices for fast merging
        self._user_sparse_col_idx = data_info.user_sparse_col.index if data_info.user_sparse_col.index else []
        self._item_sparse_col_idx = data_info.item_sparse_col.index if data_info.item_sparse_col.index else []
        self._user_dense_col_idx = data_info.user_dense_col.index if data_info.user_dense_col.index else []
        self._item_dense_col_idx = data_info.item_dense_col.index if data_info.item_dense_col.index else []

        # Pre-compute merge reindex arrays for fast column reordering
        self._sparse_reindex = None
        self._dense_reindex = None
        if self._user_sparse_col_idx and self._item_sparse_col_idx:
            orig_cols = list(self._user_sparse_col_idx) + list(self._item_sparse_col_idx)
            self._sparse_reindex = np.argsort(orig_cols)
        if self._user_dense_col_idx and self._item_dense_col_idx:
            orig_cols = list(self._user_dense_col_idx) + list(self._item_dense_col_idx)
            self._dense_reindex = np.argsort(orig_cols)

    def _build_col_offset(self):
        """Build mapping from column name to its offset in the embedding table."""
        col_offset = {}
        if self.sparse_offset is not None:
            all_sparse_cols = []
            if self.sparse_col:
                all_sparse_cols.extend(self.sparse_col)
            if self.multi_sparse_col:
                for field in self.multi_sparse_col:
                    all_sparse_cols.extend(field)
            for i, col in enumerate(all_sparse_cols):
                if i < len(self.sparse_offset):
                    col_offset[col] = int(self.sparse_offset[i])
        return col_offset

    def _build_lookup_tables(self, user_features_df, item_features_df):
        """Pre-compute numpy lookup tables for O(1) feature access."""
        # Build user lookup tables
        if user_features_df is not None:
            if self.user_sparse_col:
                self._user_sparse_table = self._build_sparse_table(
                    user_features_df, "user", self.user_unique_vals,
                    self.user_sparse_col, self.n_users
                )
            if self.user_dense_col:
                self._user_dense_table = self._build_dense_table(
                    user_features_df, "user", self.user_unique_vals,
                    self.user_dense_col, self.n_users
                )

        # Build item lookup tables
        if item_features_df is not None:
            if self.item_sparse_col:
                self._item_sparse_table = self._build_sparse_table(
                    item_features_df, "item", self.item_unique_vals,
                    self.item_sparse_col, self.n_items
                )
            if self.item_dense_col:
                self._item_dense_table = self._build_dense_table(
                    item_features_df, "item", self.item_unique_vals,
                    self.item_dense_col, self.n_items
                )

    def _build_sparse_table(self, df, id_col, unique_vals, col_names, n_ids):
        """Build pre-computed sparse feature table indexed by inner IDs.
        
        Uses vectorized operations for fast initialization.
        Returns numpy array of shape [n_ids, num_cols] with embedding indices.
        """
        n_cols = len(col_names)
        
        # Build reverse mapping: original_id -> inner_id using numpy for speed
        orig_to_inner = pd.Series(np.arange(len(unique_vals)), index=unique_vals)
        
        # Deduplicate - keep last occurrence
        df_dedup = df.drop_duplicates(subset=[id_col], keep="last")
        
        # Filter to only IDs that exist in our unique_vals
        valid_mask = df_dedup[id_col].isin(unique_vals)
        df_valid = df_dedup[valid_mask]
        
        # Get inner IDs for all valid rows at once
        inner_ids = orig_to_inner.loc[df_valid[id_col].values].values
        
        # Initialize table with default OOV values for each column
        table = np.zeros((n_ids, n_cols), dtype=np.int32)
        for j, col in enumerate(col_names):
            offset = self._col_offset.get(col, 0)
            oov_val = self._get_oov_for_col(col)
            table[:, j] = oov_val + offset
        
        # Vectorized assignment for each column
        for j, col in enumerate(col_names):
            if col not in df_valid.columns:
                continue
            
            offset = self._col_offset.get(col, 0)
            oov_val = self._get_oov_for_col(col)
            idx_mapping = self._get_idx_mapping_for_col(col)
            
            if idx_mapping is None:
                continue
            
            col_vals = df_valid[col].values
            
            # Vectorized mapping using numpy
            mapped_indices = np.array([
                idx_mapping.get(v, oov_val) + offset for v in col_vals
            ], dtype=np.int32)
            
            # Vectorized assignment
            table[inner_ids, j] = mapped_indices
        
        return table

    def _build_dense_table(self, df, id_col, unique_vals, col_names, n_ids):
        """Build pre-computed dense feature table indexed by inner IDs.
        
        Uses vectorized operations for fast initialization.
        Returns numpy array of shape [n_ids, num_cols] with feature values.
        """
        n_cols = len(col_names)
        table = np.zeros((n_ids, n_cols), dtype=np.float32)
        
        # Build reverse mapping using pandas Series for fast lookup
        orig_to_inner = pd.Series(np.arange(len(unique_vals)), index=unique_vals)
        
        # Deduplicate
        df_dedup = df.drop_duplicates(subset=[id_col], keep="last")
        
        # Filter to valid IDs
        valid_mask = df_dedup[id_col].isin(unique_vals)
        df_valid = df_dedup[valid_mask]
        
        if len(df_valid) == 0:
            return table
        
        # Get inner IDs for all valid rows at once
        inner_ids = orig_to_inner.loc[df_valid[id_col].values].values
        
        # Vectorized assignment for each column
        for j, col in enumerate(col_names):
            if col not in df_valid.columns:
                continue
            
            col_vals = df_valid[col].values.astype(np.float32)
            table[inner_ids, j] = col_vals
        
        return table

    def _get_oov_for_col(self, col):
        """Get OOV index for a column."""
        if col in self._sparse_oov:
            return self._sparse_oov[col]
        # For multi-sparse columns, find the field
        field_name = self._find_multi_sparse_field(col)
        if field_name and field_name in self._sparse_oov:
            return self._sparse_oov[field_name]
        return 0

    def _get_idx_mapping_for_col(self, col):
        """Get index mapping dict for a column."""
        if col in self._sparse_idx_mapping:
            return self._sparse_idx_mapping[col]
        # For multi-sparse columns, find the field
        field_name = self._find_multi_sparse_field(col)
        if field_name and field_name in self._sparse_idx_mapping:
            return self._sparse_idx_mapping[field_name]
        return None

    def _find_multi_sparse_field(self, col):
        """Find the field name for a multi-sparse column."""
        if self.multi_sparse_col:
            for field in self.multi_sparse_col:
                if col in field:
                    return field[0]
        return None

    def get_features_for_batch(
        self,
        user_ids_inner,
        item_ids_inner,
        need_user_sparse=True,
        need_user_dense=True,
        need_item_sparse=True,
        need_item_dense=True,
    ):
        """Get features for a batch using fast numpy indexing.

        This is O(batch_size) using pre-computed lookup tables.
        """
        # Fast numpy indexing: table[inner_ids] returns [batch_size, num_cols]
        user_sparse_feats = None
        if need_user_sparse and self._user_sparse_table is not None:
            user_sparse_feats = self._user_sparse_table[user_ids_inner]

        item_sparse_feats = None
        if need_item_sparse and self._item_sparse_table is not None:
            item_sparse_feats = self._item_sparse_table[item_ids_inner]

        user_dense_feats = None
        if need_user_dense and self._user_dense_table is not None:
            user_dense_feats = self._user_dense_table[user_ids_inner]

        item_dense_feats = None
        if need_item_dense and self._item_dense_table is not None:
            item_dense_feats = self._item_dense_table[item_ids_inner]

        # Merge features using pre-computed reindex arrays
        sparse_indices = self._merge_sparse_fast(user_sparse_feats, item_sparse_feats)
        dense_values = self._merge_dense_fast(user_dense_feats, item_dense_feats)

        return sparse_indices, dense_values

    def _merge_sparse_fast(self, user_feats, item_feats):
        """Fast merge of user and item sparse features."""
        if user_feats is not None and item_feats is not None:
            concat = np.concatenate([user_feats, item_feats], axis=1)
            if self._sparse_reindex is not None:
                return concat[:, self._sparse_reindex]
            return concat
        elif user_feats is not None:
            return user_feats
        elif item_feats is not None:
            return item_feats
        return None

    def _merge_dense_fast(self, user_feats, item_feats):
        """Fast merge of user and item dense features."""
        if user_feats is not None and item_feats is not None:
            concat = np.concatenate([user_feats, item_feats], axis=1)
            if self._dense_reindex is not None:
                return concat[:, self._dense_reindex]
            return concat
        elif user_feats is not None:
            return user_feats
        elif item_feats is not None:
            return item_feats
        return None

    def get_item_features_by_inner_ids(self, item_ids_inner):
        """Get item features for sampled items using fast numpy indexing."""
        item_sparse = None
        if self._item_sparse_table is not None:
            item_sparse = self._item_sparse_table[item_ids_inner]

        item_dense = None
        if self._item_dense_table is not None:
            item_dense = self._item_dense_table[item_ids_inner]

        return item_sparse, item_dense

    def get_user_features_by_inner_ids(self, user_ids_inner):
        """Get user features for users using fast numpy indexing."""
        user_sparse = None
        if self._user_sparse_table is not None:
            user_sparse = self._user_sparse_table[user_ids_inner]

        user_dense = None
        if self._user_dense_table is not None:
            user_dense = self._user_dense_table[user_ids_inner]

        return user_sparse, user_dense


class LazyCollator(BaseCollator):
    """High-performance collator using pre-computed numpy lookup tables.

    Uses O(1) numpy array indexing for feature lookup instead of DataFrame operations.
    """

    def __init__(
        self,
        model,
        data_info,
        train_data,
        backend,
        separate_features=False,
        temperature=0.75,
    ):
        super().__init__(model, data_info, backend, separate_features, temperature)
        self.feature_joiner = LazyFeatureJoiner(
            data_info,
            train_data,
            train_data.user_features_df,
            train_data.item_features_df,
        )
        # Pre-compute column indices as numpy arrays for fast slicing
        self._user_sparse_idx = np.array(self.user_sparse_col_index) if self.user_sparse_col_index else None
        self._item_sparse_idx = np.array(self.item_sparse_col_index) if self.item_sparse_col_index else None
        self._user_dense_idx = np.array(self.user_dense_col_index) if self.user_dense_col_index else None
        self._item_dense_idx = np.array(self.item_dense_col_index) if self.item_dense_col_index else None

    def __call__(self, batch):
        user_indices = batch["user"]
        item_indices = batch["item"]
        labels = batch["label"]

        # Fast feature lookup using numpy indexing
        sparse_batch, dense_batch = self.feature_joiner.get_features_for_batch(
            user_indices, item_indices
        )

        # Handle separate features mode with pre-computed indices
        if self.separate_features:
            sparse_batch = self._split_sparse_features(sparse_batch)
            dense_batch = self._split_dense_features(dense_batch)

        seq_batch = self.get_seqs(user_indices, item_indices)

        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch

        return batch_cls(
            users=user_indices,
            items=item_indices,
            labels=labels,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )

    def _split_sparse_features(self, sparse_batch):
        """Split sparse features into user/item using pre-computed indices."""
        if sparse_batch is None:
            return None
        user_sparse = sparse_batch[:, self._user_sparse_idx] if self._user_sparse_idx is not None else None
        item_sparse = sparse_batch[:, self._item_sparse_idx] if self._item_sparse_idx is not None else None
        return PairFeats(user_sparse, item_sparse)

    def _split_dense_features(self, dense_batch):
        """Split dense features into user/item using pre-computed indices."""
        if dense_batch is None:
            return None
        user_dense = dense_batch[:, self._user_dense_idx] if self._user_dense_idx is not None else None
        item_dense = dense_batch[:, self._item_dense_idx] if self._item_dense_idx is not None else None
        return PairFeats(user_dense, item_dense)


class LazyPointwiseCollator(LazyCollator):
    """High-performance lazy collator for pointwise loss with negative sampling."""

    def __init__(
        self,
        model,
        data_info,
        train_data,
        backend,
        separate_features=False,
    ):
        super().__init__(model, data_info, train_data, backend, separate_features)
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        # Pre-compute repeat factor
        self._repeat_factor = self.num_neg + 1

    def __call__(self, batch):
        batch_size = len(batch["user"])
        
        # Pre-allocate arrays for efficiency
        user_batch = np.repeat(batch["user"], self._repeat_factor)
        item_batch = np.repeat(batch["item"], self._repeat_factor)
        label_batch = np.zeros(batch_size * self._repeat_factor, dtype=np.float32)
        label_batch[::self._repeat_factor] = 1.0

        # Sample negative items
        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)
        
        # Vectorized negative item insertion
        for i in range(self.num_neg):
            item_batch[(i + 1)::self._repeat_factor] = items_neg[i::self.num_neg]

        # Fast feature lookup
        sparse_batch, dense_batch = self._get_pointwise_feats_fast(user_batch, item_batch)

        seq_batch = self.get_seqs(user_batch, item_batch)

        if self.dual_seq:
            batch_cls = PointwiseDualSeqBatch
        elif self.separate_features:
            batch_cls = PointwiseSepFeatBatch
        else:
            batch_cls = PointwiseBatch

        return batch_cls(
            users=user_batch,
            items=item_batch,
            labels=label_batch,
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )

    def _get_pointwise_feats_fast(self, user_indices, item_indices):
        """Get features using fast numpy indexing."""
        # Get all features in one call
        sparse_batch, dense_batch = self.feature_joiner.get_features_for_batch(
            user_indices, item_indices
        )

        if self.separate_features:
            sparse_batch = self._split_sparse_features(sparse_batch)
            dense_batch = self._split_dense_features(dense_batch)

        return sparse_batch, dense_batch


class LazyPairwiseCollator(LazyCollator):
    """High-performance lazy collator for pairwise loss with negative sampling."""

    def __init__(
        self,
        model,
        data_info,
        train_data,
        backend,
        repeat_positives,
    ):
        super().__init__(model, data_info, train_data, backend, separate_features=True)
        self.sampler = model.sampler
        self.num_neg = model.num_neg
        self.repeat_positives = repeat_positives

    def __call__(self, batch):
        if self.repeat_positives and self.num_neg > 1:
            users = np.repeat(batch["user"], self.num_neg)
            items_pos = np.repeat(batch["item"], self.num_neg)
        else:
            users = batch["user"]
            items_pos = batch["item"]

        items_neg = self.sample_neg_items(batch, self.sampler, self.num_neg)

        # Fast feature lookup using numpy indexing
        sparse_batch, dense_batch = self._get_pairwise_feats_fast(users, items_pos, items_neg)

        seq_batch = self.get_seqs(users, items_pos)
        if self.has_seq and not self.repeat_positives and self.num_neg > 1:
            seq_batch = seq_batch.repeat(self.num_neg)

        return PairwiseBatch(
            queries=users,
            item_pairs=(items_pos, items_neg),
            sparse_indices=sparse_batch,
            dense_values=dense_batch,
            seqs=seq_batch,
            backend=self.backend,
        )

    def _get_pairwise_feats_fast(self, users, items_pos, items_neg):
        """Get features using fast numpy indexing."""
        # Get user features directly from lookup table
        user_sparse, user_dense = self.feature_joiner.get_user_features_by_inner_ids(users)

        # Get positive item features
        item_pos_sparse, item_pos_dense = self.feature_joiner.get_item_features_by_inner_ids(items_pos)

        # Get negative item features
        item_neg_sparse, item_neg_dense = self.feature_joiner.get_item_features_by_inner_ids(items_neg)

        # Build TripleFeats
        sparse_batch = None
        if user_sparse is not None or item_pos_sparse is not None:
            sparse_batch = TripleFeats(
                query_feats=user_sparse,
                item_pos_feats=item_pos_sparse,
                item_neg_feats=item_neg_sparse,
            )

        dense_batch = None
        if user_dense is not None or item_pos_dense is not None:
            dense_batch = TripleFeats(
                query_feats=user_dense,
                item_pos_feats=item_pos_dense,
                item_neg_feats=item_neg_dense,
            )

        return sparse_batch, dense_batch
