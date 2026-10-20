import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from typing import List
import dgl.nn as dglnn

class KDE(nn.Module):
    """
    Multi-scale Gaussian KDE with learnable per-bandwidth weights (softmax).
    """
    def __init__(self, train_dist_matrix: torch.Tensor, bandwidths=[1.0]):
        super().__init__()
        self.register_buffer("train_dist_matrix", train_dist_matrix)
        self.N = train_dist_matrix.size(0)  # number of reference samples
        self.bandwidths = bandwidths
        self.logits = nn.Parameter(torch.ones(len(self.bandwidths))/len(self.bandwidths))
        self.dimension = 1

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        alpha = F.softmax(self.logits, dim=0)   # shape (K,)
        M, N = dist_matrix.size()
        total_kde = torch.zeros(M, device=dist_matrix.device, dtype=dist_matrix.dtype)
        for k, bw in enumerate(self.bandwidths):
            exponent = -0.5 * (dist_matrix / bw)**2
            kernel_vals = torch.exp(exponent)
            sum_over_ref = kernel_vals.sum(dim=1)
            gauss_factor = 1.0 / ((2*math.pi*(bw**2))**(0.5*self.dimension))
            scale_factor = (1.0 / self.N) * gauss_factor
            partial_kde = scale_factor * sum_over_ref
            total_kde += alpha[k] * partial_kde
        return total_kde

    def compute_kde_score_train(self) -> torch.Tensor:
        return self.forward(self.train_dist_matrix)

    def compute_kde_score_test(self, test_to_train_dist_matrix: torch.Tensor) -> torch.Tensor:
        return self.forward(test_to_train_dist_matrix)


##############################################################################
#                          DGMMD                             #
##############################################################################

class DGMMD(nn.Module):
    """
    A GNN-based model that computes pairwise graph distances using
    an MMD-based kernel approach.
    """
    def __init__(
        self, 
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        bandwidths: List[float] = [1.0],
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()

        self.bandwidths = bandwidths
        
        # GNN Layers
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(dglnn.GraphConv(in_dim, out_dim, norm='both', bias=True))
            if batch_norm:
                self.bns = nn.ModuleList([nn.BatchNorm1d(out_dim)])
            else:
                self.bns = None
        else:
            self.layers.append(dglnn.GraphConv(in_dim, hidden_dim, norm='both', bias=True))
            for _ in range(num_layers - 2):
                self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, norm='both', bias=True))
            self.layers.append(dglnn.GraphConv(hidden_dim, out_dim, norm='both', bias=True))

            if batch_norm:
                self.bns = nn.ModuleList(
                    [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers - 1)] 
                    + [nn.BatchNorm1d(out_dim)]
                )
            else:
                self.bns = None
                       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, g: dgl.DGLGraph, h: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN over a single (batched) DGLGraph
        with node features h.
        """
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if self.bns is not None:
                h = self.bns[i](h)
            if i != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def compute_distance_matrix(self, graphs, graphs_b=None):
        """
        Compute graph distances with node-level distribution MMD.
        """
        return self._compute_distance_matrix_node_level(graphs, graphs_b)

    def _compute_distance_matrix_node_level(self, graphs, graphs_b=None):
        """

        """
        node_embeddings_A, graph_ids_A, num_graphs_A = self._compute_embeddings_node_level(graphs)
        dist_sq_A = self._pairwise_distances(node_embeddings_A)

        # Precompute indices for each graph in A
        graph_indices_A = [(graph_ids_A == i).nonzero(as_tuple=True)[0]
                           for i in range(num_graphs_A)]

        if graphs_b is None:
            # Intra-set (N x N)
            distance_matrix = torch.zeros(num_graphs_A, num_graphs_A, device=node_embeddings_A.device)
            for i in range(num_graphs_A):
                idx_i = graph_indices_A[i]
                dist_ii = dist_sq_A[idx_i][:, idx_i]
                for j in range(i, num_graphs_A):
                    if j == i:
                        distance_matrix[i, j] = 0.0
                        continue
                    idx_j = graph_indices_A[j]
                    dist_jj = dist_sq_A[idx_j][:, idx_j]
                    dist_ij = dist_sq_A[idx_i][:, idx_j]
                    mmd_ij = self._compute_mmd_sup(dist_ii, dist_jj, dist_ij)
                    distance_matrix[i, j] = distance_matrix[j, i] = mmd_ij
            return distance_matrix
        else:
            # Cross-set (N x M)
            node_embeddings_B, graph_ids_B, num_graphs_B = self._compute_embeddings_node_level(graphs_b)
            dist_sq_B = self._pairwise_distances(node_embeddings_B)
            dist_sq_cross = self._pairwise_distances_cross(node_embeddings_A, node_embeddings_B)

            graph_indices_B = [(graph_ids_B == j).nonzero(as_tuple=True)[0]
                               for j in range(num_graphs_B)]

            distance_matrix = torch.zeros(num_graphs_A, num_graphs_B, device=node_embeddings_A.device)
            for i in range(num_graphs_A):
                idx_i = graph_indices_A[i]
                dist_ii = dist_sq_A[idx_i][:, idx_i]
                for j in range(num_graphs_B):
                    idx_j = graph_indices_B[j]
                    dist_jj = dist_sq_B[idx_j][:, idx_j]
                    dist_ij = dist_sq_cross[idx_i][:, idx_j]
                    mmd_ij = self._compute_mmd_sup(dist_ii, dist_jj, dist_ij)
                    distance_matrix[i, j] = mmd_ij
            return distance_matrix

    def _compute_embeddings_node_level(self, graphs):
        """

        """
        if isinstance(graphs, list):
            batched_graph = dgl.batch(graphs)
            num_graphs = len(graphs)
        else:
            batched_graph = graphs
            num_graphs = len(batched_graph.batch_num_nodes())

        h = batched_graph.ndata['attr']
        h = self.forward(batched_graph, h)

        # Build graph_ids
        num_nodes_per_graph = batched_graph.batch_num_nodes().tolist()
        graph_ids_list = []
        for i, n in enumerate(num_nodes_per_graph):
            graph_ids_list.extend([i]*n)
        graph_ids = torch.tensor(graph_ids_list, device=h.device)
        return h, graph_ids, num_graphs

    def _pairwise_distances(self, embeddings):
        """
        Compute pairwise squared Euclidean distances among the embeddings.
        embeddings: (N, emb_dim)
        Returns: (N, N)
        """
        norm = (embeddings ** 2).sum(dim=1, keepdim=True)
        dist_sq = norm + norm.t() - 2.0 * torch.mm(embeddings, embeddings.t())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return dist_sq

    def _pairwise_distances_cross(self, embA, embB):
        """
        embA: [N, d], embB: [M, d]
        Returns [N, M]
        """
        normA = (embA ** 2).sum(dim=1, keepdim=True)  # [N,1]
        normB = (embB ** 2).sum(dim=1, keepdim=True)  # [M,1]
        dist_sq_cross = normA + normB.t() - 2.0 * (embA @ embB.t())
        dist_sq_cross = torch.clamp(dist_sq_cross, min=0.0)
        return dist_sq_cross

    def _compute_mmd_sup(self, dist_ii, dist_jj, dist_ij):
        """

        """
        device = dist_ii.device
        mmd_sup = torch.zeros(1, device=device)
        
        for bw in self.bandwidths:
            K_ii = torch.exp(-dist_ii / bw).mean()
            K_jj = torch.exp(-dist_jj / bw).mean()
            K_ij = torch.exp(-dist_ij / bw).mean()
            mmd_sq = K_ii + K_jj - 2 * K_ij
            mmd_sq = torch.clamp(mmd_sq, min=1e-8)
            mmd = torch.sqrt(mmd_sq)
            mmd_sup = torch.max(mmd_sup, mmd)
        return mmd_sup


##############################################################################

##############################################################################

class LGKDE(nn.Module):
    """



    """
    def __init__(
        self, 
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        bandwidths: List[float] = [1.0],
        dropout: float = 0.1,
        batch_norm: bool = True,
        kde_dimension: int = 1,
        learn_kde_weights: bool = True
    ):
        super().__init__()
        
        # Core DGMMD model.
        self.dgmmd = DGMMD(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            bandwidths=bandwidths,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
        # Learnable weights for multi-scale KDE
        self.kde_logits = nn.Parameter(torch.ones(len(bandwidths))/len(bandwidths))
        self.kde_dimension = kde_dimension
        
        # Optionally freeze KDE weights
        if not learn_kde_weights:
            self.kde_logits.requires_grad_(False)

    def compute_kde_scores(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute KDE scores from distance matrix using learned bandwidth weights.
        
        Args:
            dist_matrix: Shape (M, N) pairwise distances
        Returns:
            torch.Tensor: Shape (M,) KDE scores
        """
        alpha = F.softmax(self.kde_logits, dim=0)
        M, N = dist_matrix.size()
        
        total_kde = torch.zeros(M, device=dist_matrix.device, dtype=dist_matrix.dtype)
        for k, bw in enumerate(self.dgmmd.bandwidths):
            exponent = -0.5 * (dist_matrix / bw)**2
            kernel_vals = torch.exp(exponent)
            partial_kde = (1.0 / N) * kernel_vals.sum(dim=1) / ((2*math.pi*(bw**2)) ** (0.5*self.kde_dimension))
            total_kde += alpha[k] * partial_kde
            
        return total_kde

    def get_reference_scores(self, reference_graphs: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute density scores for reference set (usually training graphs).
        
        Args:
            reference_graphs: Batched reference graphs
        Returns:
            torch.Tensor: Density scores for reference graphs
        """
        ref_dist = self.dgmmd.compute_distance_matrix(reference_graphs, graphs_b=None)
        return self.compute_kde_scores(ref_dist)
    
    def get_query_scores(self, query_graphs: dgl.DGLGraph, reference_graphs: dgl.DGLGraph) -> torch.Tensor:
        """
        Compute density scores for query graphs relative to reference graphs.
        
        Args:
            query_graphs: Batched query graphs
            reference_graphs: Batched reference graphs
        Returns:
            torch.Tensor: Density scores for query graphs
        """
        query_dist = self.dgmmd.compute_distance_matrix(query_graphs, graphs_b=reference_graphs)
        return self.compute_kde_scores(query_dist)

    @torch.no_grad()
    def get_anomaly_scores(
        self,
        test_graphs: dgl.DGLGraph,
        train_graphs: dgl.DGLGraph,
        threshold_percentile: float = 10
    ):
        """
        Compute anomaly scores and threshold for test graphs.
        
        Args:
            test_graphs: Batched test graphs
            train_graphs: Batched training graphs
            threshold_percentile: Percentile for anomaly threshold
        Returns:
            (test_scores, train_scores, predictions, threshold)
        """
        self.eval()
        
        # Compute scores
        train_scores = self.get_reference_scores(train_graphs)
        test_scores = self.get_query_scores(test_graphs, train_graphs)
        
        # Compute threshold from training scores
        threshold = torch.quantile(train_scores, threshold_percentile/100)
        predictions = (test_scores <= threshold).int()
        
        return test_scores, train_scores, predictions, threshold
    
    
if __name__ == "__main__":
    print("LGKDE model definition.")
    pass
