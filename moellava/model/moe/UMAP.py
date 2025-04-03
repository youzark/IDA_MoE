import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import curve_fit
import numpy as np
from typing import Tuple, Optional

class ParametricUMAP(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_components: int = 2,
        negative_sample_rate: int = 5,
        device: str = 'cuda'
    ):
        super().__init__()
        self.encoder = encoder
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.negative_sample_rate = negative_sample_rate
        self.device = device
        self.a, self.b = self._find_ab_params(min_dist, spread)

    @staticmethod
    def _find_ab_params(min_dist: float, spread: float) -> Tuple[float, float]:
        """Compute a, b params for the UMAP loss"""
        def curve(x, a, b):
            return 1.0 / (1.0 + a * x ** (2 * b))
        
        xv = torch.linspace(0, spread * 3, 300)
        yv = torch.zeros_like(xv)
        yv[xv < min_dist] = 1.0
        yv[xv >= min_dist] = torch.exp(-(xv[xv >= min_dist] - min_dist) / spread)
        
        params = torch.tensor(curve_fit(curve, xv.numpy(), yv.numpy())[0])
        return params[0].item(), params[1].item()

    def _smooth_knn_dist(
        self,
        distances: torch.Tensor,
        k: int,
        n_iter: int = 64,
        local_connectivity: float = 1.0,
        bandwidth: float = 1.0,
        min_k_dist_scale: float = 1e-3,
        tolerance: float = 1e-5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized version of smooth_knn_dist
        """
        target = np.log2(k) * bandwidth
        
        rho = distances[:, local_connectivity - 1]
        
        # Binary search for sigma
        lo = torch.zeros_like(rho)
        hi = torch.ones_like(rho) * float('inf')
        mid = torch.ones_like(rho)
        
        for _ in range(n_iter):
            # Compute psum using broadcasting
            d = distances - rho.unsqueeze(1)  # [n_samples, n_neighbors]
            exp_d = torch.where(d > 0, torch.exp(-d / mid.unsqueeze(1)), torch.ones_like(d))
            psum = torch.sum(exp_d, dim=1)  # [n_samples]
            
            # Update bounds
            too_high = psum > target
            too_low = psum < target
            within_tolerance = torch.abs(psum - target) < tolerance
            
            if torch.all(within_tolerance):
                break
                
            # Update hi, lo, and mid
            hi = torch.where(too_high, mid, hi)
            lo = torch.where(too_low, mid, lo)
            
            mid = torch.where(
                too_high,
                (lo + hi) / 2.0,
                torch.where(hi == float('inf'), mid * 2, (lo + hi) / 2.0)
            )
        
        result = mid
        
        # Scale results
        mean_distances = torch.mean(distances, dim=1)  # [n_samples]
        min_threshold = torch.where(
            rho > 0.0,
            min_k_dist_scale * mean_distances,
            min_k_dist_scale * torch.mean(distances)
        )
        
        result = torch.maximum(result, min_threshold)
        
        return result.unsqueeze(1), rho.unsqueeze(1)

    def topk_neighbor_distances(
        self,
        input,
        k,
    ):
        seq_len ,_ = input.shape
        distances = torch.cdist(input,input)
        
        mask = torch.eye(distances.shape[0], device=distances.device)
        distances = distances * (1 - mask) + torch.inf * mask

        # Get kNN structure for each sequence
        k_distances, k_indices = torch.topk(
            -distances, 
            k=min(k, seq_len),
            dim=-1, 
            largest=False
        )
        k_distances = -k_distances  # [batch_size* seq_len, n_neighbors]
        k_indices = k_indices
        return k_distances, k_indices


    def fuzzy_simplicial_set(
        self,
        input,
        n_neighbors,
        local_connectivity=1,
    ):
        k_distances, k_indices = self.topk_neighbor_distances(
            input = input,
            k = n_neighbors,
        )
        sigmas, rhos = self._smooth_knn_dist(
            distances = k_distances,
            k = n_neighbors,
            local_connectivity= local_connectivity,
        )
        

    def compute_loss(self, input: torch.Tensor) -> torch.Tensor:
        """
        Compute UMAP loss for a batch of token embeddings
        Args:
            X: token embeddings [batch_size* seq_len, hidden_dim]
        Returns:
            loss: UMAP loss for this batch
        """
        
        # Get embeddings through encoder
        seq_len , dim = input.shape
        embeddings = self.encoder(input)  # [batch_size * seq_len, n_components]
        
        # Compute distances for each sequence in the batch
        high_dim_distances = torch.cdist(input, input)
        high_dim_neighbor_distance, high_dim_neighbor_indices = self.topk_neighbor_distances(
            input,
            self.n_neighbors
        )
        distances = torch.cdist(embeddings, embeddings)  # [batch_size * seq_len, batch_size * seq_len]
        
        # Get kNN structure for each sequence
        k_distances, k_indices = torch.topk(
            -distances, k=min(self.n_neighbors+1, seq_len),
            dim=-1, largest=False
        )
        k_distances = -k_distances[:,1:]  # [batch_size* seq_len, n_neighbors]
        k_indices = k_indices[:,1:]
        
        rho, sigma = self._smooth_knn_dist(k_distances)
        
        # Compute positive samples weights
        weights = torch.exp(-(k_distances - rho) / sigma)  # [batch_size, seq_len, n_neighbors]
        
        # Compute positive loss
        pos_distances = torch.gather(
            distances,
            -1,
            k_indices
        )  # [batch_size, seq_len, n_neighbors]
        
        positive_loss = torch.mean(
            weights * torch.log1p(self.a * pos_distances ** self.b)
        )
        
        # Compute negative loss using random sampling within each sequence
        n_total = seq_len
        n_negative = self.negative_sample_rate * min(self.n_neighbors, seq_len)
        
        # Generate negative samples for each sequence
        neg_indices = torch.randint(
            0, seq_len,
            (batch_size, seq_len, n_negative),
            device=X.device
        )  # [batch_size, seq_len, n_negative]
        
        # Compute negative distances
        batch_idx = torch.arange(batch_size, device=X.device)[:, None, None]
        seq_idx = torch.arange(seq_len, device=X.device)[None, :, None]
        neg_distances = distances[batch_idx, seq_idx, neg_indices]
        
        negative_loss = torch.mean(
            torch.log1p(1.0 / (1e-6 + neg_distances))
        )
        
        return positive_loss + negative_loss

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            X: token embeddings [batch_size* seq_len, hidden_dim]
        Returns:
            embeddings: [batch_size* seq_len, n_components]
        """
        embeddings = self.encoder(input)
        return embeddings
