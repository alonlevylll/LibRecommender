"""
IGMC (Inductive Graph-based Matrix Completion) PyTorch Module.

Based on the paper:
M. Zhang and Y. Chen, "Inductive Matrix Completion Based on Graph Neural Networks", ICLR 2020.
https://github.com/muhanzhang/IGMC
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IGMCModel(nn.Module):
    """
    IGMC model that uses graph neural networks to predict ratings from 
    enclosing subgraphs around (user, item) pairs.
    
    Uses a simplified relational GCN where different edge types (ratings)
    have different transformation weights via basis decomposition.
    """
    
    def __init__(
        self,
        n_relations,
        in_features=4,
        hidden_size=32,
        out_features=1,
        n_bases=4,
        n_layers=4,
        dropout_rate=0.0,
        device="cpu",
    ):
        super(IGMCModel, self).__init__()
        self.n_relations = n_relations
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.n_bases = n_bases
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.device = device
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            RGCNConv(in_features, hidden_size, n_relations, n_bases)
        )
        for _ in range(n_layers - 1):
            self.convs.append(
                RGCNConv(hidden_size, hidden_size, n_relations, n_bases)
            )
        
        # Final MLP for prediction
        self.lin1 = nn.Linear(2 * hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, out_features)
        
        self.to(device)
        
    def forward(self, data):
        """
        Forward pass for a batch of subgraphs.
        
        Parameters
        ----------
        data : dict
            Dictionary containing:
            - x: Node features [num_nodes, in_features]
            - edge_index: Edge indices [2, num_edges]
            - edge_type: Edge types [num_edges]
            - target_user_idx: Index of target user in each subgraph
            - target_item_idx: Index of target item in each subgraph
            
        Returns
        -------
        out : torch.Tensor
            Predicted ratings [batch_size]
        """
        x = data['x']
        edge_index = data['edge_index']
        edge_type = data['edge_type']
        
        # Apply GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_type)
            x = torch.tanh(x)
            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Extract user and item representations
        user_indices = data['target_user_idx']
        item_indices = data['target_item_idx']
        
        user_embeds = x[user_indices]
        item_embeds = x[item_indices]
        
        # Concatenate and predict
        concat = torch.cat([user_embeds, item_embeds], dim=1)
        out = F.relu(self.lin1(concat))
        if self.dropout_rate > 0 and self.training:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.lin2(out)
        
        return out.squeeze(-1)


class RGCNConv(nn.Module):
    """
    Relational Graph Convolutional Network layer with basis decomposition.
    Fully vectorized for GPU efficiency.
    """
    
    def __init__(self, in_channels, out_channels, num_relations, num_bases=None):
        super(RGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases if num_bases is not None else num_relations
        
        # Basis matrices
        self.bases = nn.Parameter(
            torch.Tensor(self.num_bases, in_channels, out_channels)
        )
        
        # Coefficients for each relation
        self.attn = nn.Parameter(
            torch.Tensor(num_relations, self.num_bases)
        )
        
        # Self-loop transformation
        self.root = nn.Linear(in_channels, out_channels, bias=False)
        
        # Bias
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.attn)
        nn.init.xavier_uniform_(self.root.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, edge_index, edge_type):
        """Forward pass (fully vectorized)."""
        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return self.root(x) + self.bias
        
        # Compute weight matrices: W_r = sum_b attn[r, b] * bases[b]
        weights = torch.einsum('rb,bio->rio', self.attn, self.bases)
        
        source, target = edge_index
        
        # Get source features and transform by edge type
        x_src = x[source]
        edge_weights = weights[edge_type]
        h = torch.einsum('ei,eio->eo', x_src, edge_weights)
        
        # Aggregate
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        out.index_add_(0, target, h)
        
        # Add self-loop and bias
        out = out + self.root(x) + self.bias
        
        return out


class SubgraphDataset(torch.utils.data.Dataset):
    """
    Dataset that pre-computes and stores subgraphs for fast training.
    """
    
    def __init__(
        self,
        users,
        items, 
        labels,
        user_consumed,
        item_consumed,
        ratings_matrix,
        n_users,
        n_items,
        max_neighbors=50,
    ):
        self.users = np.asarray(users)
        self.items = np.asarray(items)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.user_consumed = user_consumed
        self.item_consumed = item_consumed
        self.ratings_matrix = ratings_matrix
        self.n_users = n_users
        self.n_items = n_items
        self.max_neighbors = max_neighbors
        
        # Pre-convert consumed lists to arrays for faster indexing
        self._user_consumed_arrays = {}
        for u, items_list in user_consumed.items():
            self._user_consumed_arrays[u] = np.array(items_list, dtype=np.int64)
        
        self._item_consumed_arrays = {}
        for i, users_list in item_consumed.items():
            self._item_consumed_arrays[i] = np.array(users_list, dtype=np.int64)
        
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        user_idx = self.users[idx]
        item_idx = self.items[idx]
        label = self.labels[idx]
        
        # Fast 1-hop subgraph extraction
        sg = self._extract_subgraph_fast(user_idx, item_idx)
        sg['label'] = label
        return sg
    
    def _extract_subgraph_fast(self, user_idx, item_idx):
        """
        Fast 1-hop subgraph extraction using numpy operations.
        """
        # Get 1-hop neighbors
        user_items = self._user_consumed_arrays.get(user_idx, np.array([], dtype=np.int64))
        item_users = self._item_consumed_arrays.get(item_idx, np.array([], dtype=np.int64))
        
        # Sample if too many neighbors
        if len(user_items) > self.max_neighbors:
            user_items = np.random.choice(user_items, self.max_neighbors, replace=False)
        if len(item_users) > self.max_neighbors:
            item_users = np.random.choice(item_users, self.max_neighbors, replace=False)
        
        # Remove target item from user's items and target user from item's users
        user_items = user_items[user_items != item_idx]
        item_users = item_users[item_users != user_idx]
        
        # Build node list: target_user, target_item, neighbor_items, neighbor_users
        # Users: indices 0, then after items
        # Items: index 1, then neighbors
        
        n_neighbor_items = len(user_items)
        n_neighbor_users = len(item_users)
        
        # Node indices:
        # 0: target user
        # 1: target item  
        # 2 to 2+n_neighbor_items-1: neighbor items (connected to target user)
        # 2+n_neighbor_items onwards: neighbor users (connected to target item)
        
        num_nodes = 2 + n_neighbor_items + n_neighbor_users
        
        # Build edges
        edges_src = []
        edges_tgt = []
        edge_types = []
        
        # Edges from target user (0) to neighbor items (2 to 2+n_neighbor_items-1)
        for local_idx, neighbor_item in enumerate(user_items):
            rating = int(self.ratings_matrix[user_idx, neighbor_item])
            item_local = 2 + local_idx
            # Bidirectional
            edges_src.extend([0, item_local])
            edges_tgt.extend([item_local, 0])
            edge_types.extend([rating, rating])
        
        # Edges from target item (1) to neighbor users
        for local_idx, neighbor_user in enumerate(item_users):
            rating = int(self.ratings_matrix[neighbor_user, item_idx])
            user_local = 2 + n_neighbor_items + local_idx
            # Bidirectional
            edges_src.extend([1, user_local])
            edges_tgt.extend([user_local, 1])
            edge_types.extend([rating, rating])
        
        # Node features: [is_user, is_item, is_target_user, is_target_item]
        node_features = np.zeros((num_nodes, 4), dtype=np.float32)
        
        # Target user
        node_features[0, 0] = 1.0  # is_user
        node_features[0, 2] = 1.0  # is_target_user
        
        # Target item
        node_features[1, 1] = 1.0  # is_item
        node_features[1, 3] = 1.0  # is_target_item
        
        # Neighbor items
        for i in range(n_neighbor_items):
            node_features[2 + i, 1] = 1.0  # is_item
        
        # Neighbor users
        for i in range(n_neighbor_users):
            node_features[2 + n_neighbor_items + i, 0] = 1.0  # is_user
        
        return {
            'num_nodes': num_nodes,
            'edge_index': np.array([edges_src, edges_tgt], dtype=np.int64) if edges_src else np.zeros((2, 0), dtype=np.int64),
            'edge_type': np.array(edge_types, dtype=np.int64) if edge_types else np.zeros(0, dtype=np.int64),
            'x': node_features,
            'target_user_idx': 0,
            'target_item_idx': 1,
        }


def collate_subgraphs(batch):
    """
    Collate a batch of subgraphs into a single batched graph.
    Works with DataLoader.
    """
    batch_x = []
    batch_edge_index = []
    batch_edge_type = []
    batch_assignment = []
    target_user_indices = []
    target_item_indices = []
    labels = []
    
    node_offset = 0
    
    for batch_idx, sg in enumerate(batch):
        num_nodes = sg['num_nodes']
        
        batch_x.append(sg['x'])
        
        # Offset edge indices
        edge_idx = sg['edge_index']
        if edge_idx.shape[1] > 0:
            batch_edge_index.append(edge_idx + node_offset)
            batch_edge_type.append(sg['edge_type'])
        
        # Batch assignment
        batch_assignment.extend([batch_idx] * num_nodes)
        
        # Target indices (offset)
        target_user_indices.append(sg['target_user_idx'] + node_offset)
        target_item_indices.append(sg['target_item_idx'] + node_offset)
        
        if 'label' in sg:
            labels.append(sg['label'])
        
        node_offset += num_nodes
    
    # Concatenate
    result = {
        'x': torch.from_numpy(np.concatenate(batch_x, axis=0)).float(),
        'edge_index': torch.from_numpy(np.concatenate(batch_edge_index, axis=1)).long() if batch_edge_index else torch.zeros(2, 0, dtype=torch.long),
        'edge_type': torch.from_numpy(np.concatenate(batch_edge_type)).long() if batch_edge_type else torch.zeros(0, dtype=torch.long),
        'batch': torch.tensor(batch_assignment, dtype=torch.long),
        'target_user_idx': torch.tensor(target_user_indices, dtype=torch.long),
        'target_item_idx': torch.tensor(target_item_indices, dtype=torch.long),
    }
    
    if labels:
        result['labels'] = torch.tensor(labels, dtype=torch.float32)
    
    return result
