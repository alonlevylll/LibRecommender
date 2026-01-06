"""
IGMC (Inductive Graph-based Matrix Completion) Algorithm.

References
----------
[1] Muhan Zhang and Yixin Chen. "Inductive Matrix Completion Based on Graph Neural Networks."
    International Conference on Learning Representations (ICLR), 2020.
    https://github.com/muhanzhang/IGMC
"""

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..bases import Base, ModelMeta
from .torch_modules import IGMCModel
from .torch_modules.igmc_module import SubgraphDataset, collate_subgraphs


class IGMC(Base, metaclass=ModelMeta, backend="torch"):
    """
    IGMC (Inductive Graph-based Matrix Completion) algorithm.
    
    IGMC is a graph neural network based recommendation model that:
    1. Extracts local enclosing subgraphs around (user, item) pairs
    2. Uses a relational GCN to learn representations from subgraph structure
    3. Predicts ratings from the learned user and item embeddings
    
    Key advantages:
    - Inductive: can generalize to new users/items not seen during training
    - No global embeddings: uses only local graph structure
    - Handles different rating values as different edge types
    
    Parameters
    ----------
    task : str
        Recommendation task, either "rating" or "ranking".
    data_info : DataInfo
        Object containing data information.
    loss_type : str, default: "mse"
        Loss function type, either "mse" or "mae".
    hidden_size : int, default: 32
        Hidden layer size for GNN.
    n_layers : int, default: 4
        Number of GNN layers.
    n_bases : int, default: 4
        Number of basis matrices for relation weight decomposition.
    max_neighbors : int, default: 50
        Maximum number of neighbors to sample for each node.
    n_epochs : int, default: 20
        Number of training epochs.
    lr : float, default: 0.001
        Learning rate.
    lr_decay_step : int, default: 50
        Steps between learning rate decay.
    lr_decay_factor : float, default: 0.1
        Learning rate decay factor.
    reg : float, default: 0.0
        L2 regularization weight.
    batch_size : int, default: 50
        Training batch size.
    dropout_rate : float, default: 0.0
        Dropout rate.
    device : str, default: "cuda"
        Device to use ("cuda" or "cpu").
    seed : int, default: 42
        Random seed.
    lower_upper_bound : tuple, default: None
        Lower and upper rating bounds for clipping predictions.
        
    References
    ----------
    .. [1] Zhang, M., & Chen, Y. (2020). Inductive matrix completion based on 
       graph neural networks. In International Conference on Learning 
       Representations.
    """
    
    user_variables = []
    item_variables = []
    sparse = False
    dense = False
    
    def __init__(
        self,
        task,
        data_info,
        loss_type="mse",
        hidden_size=32,
        n_layers=4,
        n_bases=4,
        max_neighbors=50,
        n_epochs=20,
        lr=0.001,
        lr_decay_step=50,
        lr_decay_factor=0.1,
        reg=0.0,
        batch_size=50,
        dropout_rate=0.0,
        device="cuda",
        seed=42,
        lower_upper_bound=None,
    ):
        super().__init__(task, data_info, lower_upper_bound)
        
        self.all_args = locals()
        self.loss_type = loss_type
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_bases = n_bases
        self.max_neighbors = max_neighbors
        self.n_epochs = n_epochs
        self.lr = lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_factor = lr_decay_factor
        self.reg = reg
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.seed = seed
        
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"IGMC using device: {self.device}")
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Number of relations = number of unique ratings
        # For rating task, ratings are typically 1-5, so 5 relations
        # Add 1 for 0-indexed ratings
        _, max_rating = data_info.min_max_rating
        self.n_relations = int(max_rating) + 1
        
        self.model_built = False
        self.torch_model = None
        self.optimizer = None
        self.scheduler = None
        
        # Graph structures
        self.user_consumed = None
        self.item_consumed = None
        self.ratings_matrix = None
        
    def _build_consumed_dicts(self, train_data):
        """Build user->items and item->users consumed dictionaries."""
        self.user_consumed = defaultdict(list)
        self.item_consumed = defaultdict(list)
        
        users = train_data.user_indices
        items = train_data.item_indices
        
        for u, i in zip(users, items):
            self.user_consumed[u].append(i)
            self.item_consumed[i].append(u)
    
    def _build_ratings_matrix(self, train_data):
        """Build sparse ratings matrix for subgraph extraction."""
        users = train_data.user_indices
        items = train_data.item_indices
        labels = train_data.labels
        
        # Create sparse matrix
        self.ratings_matrix = csr_matrix(
            (labels, (users, items)),
            shape=(self.n_users, self.n_items),
            dtype=np.int32,
        )
        
    def _build_default_recs(self):
        """Build default recommendations based on item popularity."""
        item_counts = defaultdict(int)
        for items in self.user_consumed.values():
            for item in items:
                item_counts[item] += 1
        
        sorted_items = sorted(item_counts.keys(), key=lambda x: item_counts[x], reverse=True)
        self.default_recs = sorted_items[:100]
        
    def build_model(self):
        """Build the IGMC PyTorch model."""
        self.torch_model = IGMCModel(
            n_relations=self.n_relations,
            in_features=4,
            hidden_size=self.hidden_size,
            out_features=1,
            n_bases=self.n_bases,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
            device=self.device,
        )
        
        self.optimizer = Adam(
            self.torch_model.parameters(),
            lr=self.lr,
            weight_decay=self.reg,
        )
        
        self.scheduler = StepLR(
            self.optimizer,
            step_size=self.lr_decay_step,
            gamma=self.lr_decay_factor,
        )
        
    def fit(
        self,
        train_data,
        neg_sampling=False,
        verbose=1,
        shuffle=True,
        eval_data=None,
        metrics=None,
        k=10,
        eval_batch_size=8192,
        eval_user_num=None,
        num_workers=4,
        early_stop=None,
    ):
        """Fit IGMC model on training data.
        
        Parameters
        ----------
        train_data : TransformedSet
            Training data.
        neg_sampling : bool, default: False
            Not used for IGMC (kept for API compatibility).
        verbose : int, default: 1
            Verbosity level.
        shuffle : bool, default: True
            Whether to shuffle training data.
        eval_data : TransformedSet, default: None
            Evaluation data.
        metrics : list, default: None
            Evaluation metrics.
        k : int, default: 10
            Parameter for ranking metrics.
        eval_batch_size : int, default: 8192
            Batch size for evaluation.
        eval_user_num : int, default: None
            Number of users for evaluation.
        num_workers : int, default: 4
            Number of DataLoader workers for parallel data loading.
        early_stop : int, default: None
            Stop training if no improvement for this many epochs.
        """
        self.show_start_time()
        
        # Build graph structures
        self._build_consumed_dicts(train_data)
        self._build_ratings_matrix(train_data)
        self._build_default_recs()
        
        if not self.model_built:
            self.build_model()
            self.model_built = True
        
        # Create dataset and dataloader
        dataset = SubgraphDataset(
            users=train_data.user_indices,
            items=train_data.item_indices,
            labels=train_data.labels,
            user_consumed=self.user_consumed,
            item_consumed=self.item_consumed,
            ratings_matrix=self.ratings_matrix,
            n_users=self.n_users,
            n_items=self.n_items,
            max_neighbors=self.max_neighbors,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_subgraphs,
            pin_memory=(self.device.type == "cuda"),
        )
        
        best_metric = float('inf') if self.task == "rating" else 0
        patience_counter = 0
        
        for epoch in range(1, self.n_epochs + 1):
            self.torch_model.train()
            total_loss = 0.0
            n_batches = len(dataloader)
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=(verbose <= 0))
            
            for batch_data in pbar:
                # Move to device
                batch_data = {k: v.to(self.device, non_blocking=True) for k, v in batch_data.items()}
                
                # Forward pass
                self.optimizer.zero_grad()
                preds = self.torch_model(batch_data)
                
                # Loss
                targets = batch_data['labels']
                if self.loss_type == "mse":
                    loss = F.mse_loss(preds, targets)
                else:
                    loss = F.l1_loss(preds, targets)
                    
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Learning rate decay
            self.scheduler.step()
            
            avg_loss = total_loss / n_batches
            if verbose >= 1:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
            
            # Evaluation
            if eval_data is not None and metrics is not None:
                eval_result = self._evaluate(
                    eval_data, metrics, k, eval_batch_size, eval_user_num
                )
                if verbose >= 1:
                    print(f"Evaluation: {eval_result}")
                
                # Early stopping
                if early_stop is not None:
                    if self.task == "rating":
                        current_metric = eval_result.get("rmse", eval_result.get("mae", avg_loss))
                        improved = current_metric < best_metric
                    else:
                        current_metric = eval_result.get("ndcg", eval_result.get("map", 0))
                        improved = current_metric > best_metric
                    
                    if improved:
                        best_metric = current_metric
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stop:
                            print(f"Early stopping at epoch {epoch}")
                            break
        
        self.print_metrics(self.task)
        return self
    
    def predict(self, user, item, cold_start="average", inner_id=False):
        """
        Predict rating for a (user, item) pair.
        
        Parameters
        ----------
        user : int or array-like
            User ID(s).
        item : int or array-like
            Item ID(s).
        cold_start : str, default: "average"
            Strategy for cold start.
        inner_id : bool, default: False
            Whether the IDs are inner IDs.
            
        Returns
        -------
        preds : float or np.ndarray
            Predicted rating(s).
        """
        self.torch_model.eval()
        
        user = np.atleast_1d(user)
        item = np.atleast_1d(item)
        
        if not inner_id:
            user = np.array([self.data_info.user2id.get(u, -1) for u in user])
            item = np.array([self.data_info.item2id.get(i, -1) for i in item])
        
        preds = []
        with torch.no_grad():
            for u, i in zip(user, item):
                if u == -1 or i == -1:
                    preds.append(self.default_prediction)
                    continue
                
                # Extract subgraph
                sg = self._extract_single_subgraph(u, i)
                batch_data = collate_subgraphs([sg])
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                
                pred = self.torch_model(batch_data).item()
                preds.append(pred)
        
        preds = np.array(preds)
        preds = self.clip_pred(preds)
        
        return preds[0] if len(preds) == 1 else preds
    
    def _extract_single_subgraph(self, user_idx, item_idx):
        """Extract a single subgraph for prediction."""
        user_items = list(self.user_consumed.get(user_idx, []))
        item_users = list(self.item_consumed.get(item_idx, []))
        
        # Sample if too many
        if len(user_items) > self.max_neighbors:
            user_items = np.random.choice(user_items, self.max_neighbors, replace=False).tolist()
        if len(item_users) > self.max_neighbors:
            item_users = np.random.choice(item_users, self.max_neighbors, replace=False).tolist()
        
        # Remove target
        user_items = [i for i in user_items if i != item_idx]
        item_users = [u for u in item_users if u != user_idx]
        
        n_neighbor_items = len(user_items)
        n_neighbor_users = len(item_users)
        num_nodes = 2 + n_neighbor_items + n_neighbor_users
        
        edges_src = []
        edges_tgt = []
        edge_types = []
        
        for local_idx, neighbor_item in enumerate(user_items):
            rating = int(self.ratings_matrix[user_idx, neighbor_item])
            item_local = 2 + local_idx
            edges_src.extend([0, item_local])
            edges_tgt.extend([item_local, 0])
            edge_types.extend([rating, rating])
        
        for local_idx, neighbor_user in enumerate(item_users):
            rating = int(self.ratings_matrix[neighbor_user, item_idx])
            user_local = 2 + n_neighbor_items + local_idx
            edges_src.extend([1, user_local])
            edges_tgt.extend([user_local, 1])
            edge_types.extend([rating, rating])
        
        node_features = np.zeros((num_nodes, 4), dtype=np.float32)
        node_features[0, 0] = 1.0
        node_features[0, 2] = 1.0
        node_features[1, 1] = 1.0
        node_features[1, 3] = 1.0
        for i in range(n_neighbor_items):
            node_features[2 + i, 1] = 1.0
        for i in range(n_neighbor_users):
            node_features[2 + n_neighbor_items + i, 0] = 1.0
        
        return {
            'num_nodes': num_nodes,
            'edge_index': np.array([edges_src, edges_tgt], dtype=np.int64) if edges_src else np.zeros((2, 0), dtype=np.int64),
            'edge_type': np.array(edge_types, dtype=np.int64) if edge_types else np.zeros(0, dtype=np.int64),
            'x': node_features,
            'target_user_idx': 0,
            'target_item_idx': 1,
        }
    
    def recommend_user(self, user, n_rec=10, cold_start="average", inner_id=False):
        """
        Recommend items for a user.
        
        Parameters
        ----------
        user : int
            User ID.
        n_rec : int, default: 10
            Number of recommendations.
        cold_start : str, default: "average"
            Strategy for cold start.
        inner_id : bool, default: False
            Whether the ID is inner ID.
            
        Returns
        -------
        recs : list
            List of recommended item IDs.
        """
        self.torch_model.eval()
        
        if not inner_id:
            user_id = self.data_info.user2id.get(user, -1)
        else:
            user_id = user
            
        if user_id == -1:
            # Cold start: return popular items
            return self.default_recs[:n_rec]
        
        # Get items user hasn't consumed
        consumed = set(self.user_consumed.get(user_id, []))
        candidates = [i for i in range(self.n_items) if i not in consumed]
        
        if not candidates:
            return []
        
        # Score all candidates in batches
        scores = []
        with torch.no_grad():
            for i in range(0, len(candidates), self.batch_size):
                batch_items = candidates[i:i + self.batch_size]
                
                subgraphs = [self._extract_single_subgraph(user_id, item) for item in batch_items]
                batch_data = collate_subgraphs(subgraphs)
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                
                batch_scores = self.torch_model(batch_data).cpu().numpy()
                scores.extend(batch_scores)
        
        # Get top-n
        scores = np.array(scores)
        top_indices = np.argsort(scores)[::-1][:n_rec]
        
        recs = [candidates[idx] for idx in top_indices]
        
        if not inner_id:
            recs = [self.data_info.id2item[i] for i in recs]
        
        return recs
    
    def _evaluate(self, eval_data, metrics, k, batch_size, user_num):
        """Evaluate the model."""
        self.torch_model.eval()
        
        users = eval_data.user_indices
        items = eval_data.item_indices
        labels = eval_data.labels
        
        # Create eval dataset
        dataset = SubgraphDataset(
            users=users,
            items=items,
            labels=labels,
            user_consumed=self.user_consumed,
            item_consumed=self.item_consumed,
            ratings_matrix=self.ratings_matrix,
            n_users=self.n_users,
            n_items=self.n_items,
            max_neighbors=self.max_neighbors,
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_subgraphs,
            pin_memory=(self.device.type == "cuda"),
        )
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_data in dataloader:
                batch_data = {k: v.to(self.device, non_blocking=True) for k, v in batch_data.items()}
                preds = self.torch_model(batch_data).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_data['labels'].cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        results = {}
        if "rmse" in metrics:
            rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
            results["rmse"] = rmse
        if "mae" in metrics:
            mae = np.mean(np.abs(all_preds - all_labels))
            results["mae"] = mae
        
        return results
    
    def save(self, path, model_name="igmc_model"):
        """
        Save model to disk.
        
        Parameters
        ----------
        path : str
            Directory path to save the model.
        model_name : str, default: "igmc_model"
            Name of the model file.
        """
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save PyTorch model
        torch.save(
            {
                'model_state_dict': self.torch_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            },
            os.path.join(path, f"{model_name}.pt")
        )
        
        # Save model attributes
        attrs = {
            'task': self.task,
            'loss_type': self.loss_type,
            'hidden_size': self.hidden_size,
            'n_layers': self.n_layers,
            'n_bases': self.n_bases,
            'max_neighbors': self.max_neighbors,
            'n_epochs': self.n_epochs,
            'lr': self.lr,
            'lr_decay_step': self.lr_decay_step,
            'lr_decay_factor': self.lr_decay_factor,
            'reg': self.reg,
            'batch_size': self.batch_size,
            'dropout_rate': self.dropout_rate,
            'seed': self.seed,
            'n_relations': self.n_relations,
            'n_users': self.n_users,
            'n_items': self.n_items,
            'user_consumed': dict(self.user_consumed),
            'item_consumed': dict(self.item_consumed),
            'default_recs': self.default_recs,
        }
        
        import pickle
        with open(os.path.join(path, f"{model_name}_attrs.pkl"), 'wb') as f:
            pickle.dump(attrs, f)
        
        # Save ratings matrix
        from scipy.sparse import save_npz
        save_npz(os.path.join(path, f"{model_name}_ratings.npz"), self.ratings_matrix)
        
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path, model_name="igmc_model", data_info=None, device="cuda"):
        """
        Load model from disk.
        
        Parameters
        ----------
        path : str
            Directory path where model is saved.
        model_name : str, default: "igmc_model"
            Name of the model file.
        data_info : DataInfo, default: None
            Data info object.
        device : str, default: "cuda"
            Device to load model to.
            
        Returns
        -------
        model : IGMC
            Loaded model.
        """
        import os
        import pickle
        from scipy.sparse import load_npz
        
        # Load attributes
        with open(os.path.join(path, f"{model_name}_attrs.pkl"), 'rb') as f:
            attrs = pickle.load(f)
        
        # Create model instance
        model = cls(
            task=attrs['task'],
            data_info=data_info,
            loss_type=attrs['loss_type'],
            hidden_size=attrs['hidden_size'],
            n_layers=attrs['n_layers'],
            n_bases=attrs['n_bases'],
            max_neighbors=attrs['max_neighbors'],
            n_epochs=attrs['n_epochs'],
            lr=attrs['lr'],
            lr_decay_step=attrs['lr_decay_step'],
            lr_decay_factor=attrs['lr_decay_factor'],
            reg=attrs['reg'],
            batch_size=attrs['batch_size'],
            dropout_rate=attrs['dropout_rate'],
            seed=attrs['seed'],
            device=device,
        )
        
        model.n_relations = attrs['n_relations']
        model.n_users = attrs['n_users']
        model.n_items = attrs['n_items']
        model.user_consumed = defaultdict(list, attrs['user_consumed'])
        model.item_consumed = defaultdict(list, attrs['item_consumed'])
        model.default_recs = attrs['default_recs']
        
        # Load ratings matrix
        model.ratings_matrix = load_npz(os.path.join(path, f"{model_name}_ratings.npz"))
        
        # Build and load PyTorch model
        model.build_model()
        model.model_built = True
        
        checkpoint = torch.load(
            os.path.join(path, f"{model_name}.pt"),
            map_location=model.device
        )
        model.torch_model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {path}")
        return model
