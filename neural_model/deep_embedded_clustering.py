"""
Deep Embedded Clustering (DEC) Implementation

This module implements DEC for driving style clustering.
DEC combines an autoencoder with a clustering loss to learn
latent representations that are better suited for clustering.

Reference: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis" (2016)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from sklearn.cluster import KMeans

from auto_encoder import AutoEncoder, MaskedMSELoss


# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class ClusteringLayer(nn.Module):
    """
    Clustering layer that converts latent space embeddings to soft cluster assignments.
    
    Uses Student's t-distribution as the kernel to measure similarity between
    embedded points and cluster centroids.
    
    Q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2) / Σ_j'(...)
    
    where α is the degrees of freedom of the t-distribution (default: 1).
    """
    
    def __init__(self, n_clusters: int, latent_dim: int, alpha: float = 1.0):
        """
        Args:
            n_clusters: Number of clusters
            latent_dim: Dimension of the latent space (input to this layer)
            alpha: Degrees of freedom for Student's t-distribution
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        self.alpha = alpha
        
        # Cluster centroids as learnable parameters
        self.cluster_centers = nn.Parameter(
            torch.zeros(n_clusters, latent_dim)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute soft assignment Q.
        
        Args:
            z: Latent representations [batch_size, latent_dim]
            
        Returns:
            q: Soft cluster assignments [batch_size, n_clusters]
        """
        # Compute squared distances: ||z_i - μ_j||²
        # z: [batch, latent_dim], centers: [n_clusters, latent_dim]
        # Result: [batch, n_clusters]
        z_expanded = z.unsqueeze(1)  # [batch, 1, latent_dim]
        centers_expanded = self.cluster_centers.unsqueeze(0)  # [1, n_clusters, latent_dim]
        
        sq_dist = torch.sum((z_expanded - centers_expanded) ** 2, dim=2)  # [batch, n_clusters]
        
        # Student's t-distribution kernel
        # q_ij = (1 + ||z_i - μ_j||² / α)^(-(α+1)/2)
        numerator = (1.0 + sq_dist / self.alpha) ** (-(self.alpha + 1.0) / 2.0)
        
        # Normalize to get soft assignments
        q = numerator / torch.sum(numerator, dim=1, keepdim=True)
        
        return q
    
    def initialize_centers(self, z: np.ndarray, n_init: int = 20) -> np.ndarray:
        """
        Initialize cluster centers using K-Means on the latent space.
        
        Args:
            z: Latent representations [n_samples, latent_dim]
            n_init: Number of K-Means initializations
            
        Returns:
            Cluster labels from K-Means
        """
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=n_init, random_state=42)
        labels = kmeans.fit_predict(z)
        
        # Copy centers to parameter
        self.cluster_centers.data = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32
        ).to(self.cluster_centers.device)
        
        return labels


class DECAutoEncoder(AutoEncoder):
    """
    Deep Embedded Clustering AutoEncoder.
    
    Extends the base AutoEncoder with:
    - Clustering layer for soft cluster assignment
    - KL divergence loss for clustering
    - Two-phase training: pretrain (reconstruction) + finetune (combined loss)
    """
    
    def __init__(
        self, 
        input_dim: int = 455, 
        latent_dim: int = 32, 
        n_clusters: int = 3,
        alpha: float = 1.0
    ):
        """
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent space
            n_clusters: Number of clusters
            alpha: Student's t-distribution degrees of freedom
        """
        super().__init__(input_dim, latent_dim)
        
        self.n_clusters = n_clusters
        self.clustering_layer = ClusteringLayer(n_clusters, latent_dim, alpha)
        self.clustering_layer.to(device)
        
        # Track training phases
        self.pretrained = False
        self.dec_trained = False
    
    def soft_assignment(self, z: torch.Tensor) -> torch.Tensor:
        """Compute soft cluster assignment Q."""
        return self.clustering_layer(z)
    
    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """
        Compute target distribution P from soft assignments Q.
        
        P is a "sharper" version of Q that emphasizes high-confidence assignments.
        
        P_ij = (Q_ij² / Σ_i Q_ij) / Σ_j'(Q_ij'² / Σ_i Q_ij')
        
        Args:
            q: Soft assignments [batch_size, n_clusters]
            
        Returns:
            p: Target distribution [batch_size, n_clusters]
        """
        # f_j = Σ_i Q_ij (soft cluster frequencies)
        f = torch.sum(q, dim=0, keepdim=True) + 1e-8
        
        # p_ij = Q_ij² / f_j
        numerator = (q ** 2) / f
        
        # Normalize
        p = numerator / torch.sum(numerator, dim=1, keepdim=True)
        
        return p
    
    def clustering_loss(self, q: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence loss between Q and P.
        
        L_clustering = KL(P || Q) = Σ_ij P_ij * log(P_ij / Q_ij)
        
        Args:
            q: Soft assignments
            p: Target distribution
            
        Returns:
            KL divergence loss
        """
        # Add small epsilon to avoid log(0)
        return torch.mean(torch.sum(p * torch.log(p / (q + 1e-8) + 1e-8), dim=1))
    
    def forward_with_clustering(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns reconstruction and cluster assignment.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, latent_z, soft_assignment_q)
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        q = self.clustering_layer(z)
        
        return x_hat, z, q
    
    def pretrain(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> None:
        """
        Phase 1: Pretrain autoencoder with reconstruction loss only.
        
        Args:
            data: Training data
            mask: Mask for padding values
            epochs: Number of pretraining epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        print("=" * 50)
        print("PHASE 1: Pretraining Autoencoder")
        print("=" * 50)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.train_model(
            optimizer=optimizer,
            epochs=epochs,
            train_data=data,
            mask=mask,
            batch_size=batch_size
        )
        
        self.pretrained = True
        print("Pretraining completed!")
    
    def initialize_clusters(self, data: np.ndarray) -> np.ndarray:
        """
        Initialize cluster centers using K-Means on encoded data.
        
        Args:
            data: Training data
            
        Returns:
            Initial cluster labels
        """
        print("\nInitializing cluster centers with K-Means...")
        
        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            z = self.encoder(data_tensor).cpu().numpy()
        
        labels = self.clustering_layer.initialize_centers(z)
        
        print(f"Initialized {self.n_clusters} cluster centers")
        print(f"Cluster distribution: {np.bincount(labels)}")
        
        return labels
    
    def train_dec(
        self,
        data: np.ndarray,
        mask: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.0001,
        update_interval: int = 1,
        reconstruction_weight: float = 0.1,
        tol: float = 0.001
    ) -> None:
        """
        Phase 2: Train with combined reconstruction + clustering loss.
        
        Args:
            data: Training data
            mask: Mask for padding values
            epochs: Number of DEC training epochs
            batch_size: Batch size
            learning_rate: Learning rate (usually lower than pretraining)
            update_interval: Epochs between target distribution updates
            reconstruction_weight: Weight α for reconstruction loss (α × L_rec + L_clust)
            tol: Tolerance for early stopping based on label change
        """
        if not self.pretrained:
            raise RuntimeError("Model must be pretrained before DEC training!")
        
        print("\n" + "=" * 50)
        print("PHASE 2: DEC Training (Clustering + Reconstruction)")
        print("=" * 50)
        print(f"Reconstruction weight: {reconstruction_weight}")
        print(f"Update interval: {update_interval} epochs")
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate
        )
        
        n_samples = len(data)
        self.losses = []
        prev_labels = None
        
        for epoch in range(epochs):
            # Compute target distribution P (using all data)
            if epoch % update_interval == 0:
                self.eval()
                with torch.no_grad():
                    data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                    z = self.encoder(data_tensor)
                    q_all = self.clustering_layer(z)
                    p_all = self.target_distribution(q_all)
                    
                    # Check for convergence
                    current_labels = torch.argmax(q_all, dim=1).cpu().numpy()
                    
                    if prev_labels is not None:
                        delta_label = np.sum(current_labels != prev_labels) / n_samples
                        print(f"Label change: {delta_label:.4f}")
                        
                        if delta_label < tol:
                            print(f"Converged! Label change < {tol}")
                            break
                    
                    prev_labels = current_labels
            
            # Training step
            self.train()
            total_loss = 0.0
            total_rec_loss = 0.0
            total_clust_loss = 0.0
            num_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_data = torch.tensor(data[batch_idx], dtype=torch.float32).to(device)
                batch_mask = torch.tensor(mask[batch_idx], dtype=torch.float32).to(device)
                batch_p = p_all[batch_idx]
                
                optimizer.zero_grad()
                
                # Forward pass
                x_hat, z, q = self.forward_with_clustering(batch_data)
                
                # Reconstruction loss
                rec_loss = self.loss_function(x_hat, batch_data, batch_mask)
                
                # Clustering loss
                clust_loss = self.clustering_loss(q, batch_p)
                
                # Combined loss
                loss = reconstruction_weight * rec_loss + clust_loss
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_rec_loss += rec_loss.item()
                total_clust_loss += clust_loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            avg_rec = total_rec_loss / num_batches
            avg_clust = total_clust_loss / num_batches
            
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Total: {avg_loss:.6f}, Rec: {avg_rec:.6f}, Clust: {avg_clust:.6f}")
            
            self.losses.append(avg_loss)
        
        self.dec_trained = True
        print("\nDEC training completed!")
    
    def predict_clusters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict cluster assignments for data.
        
        Args:
            data: Input data
            
        Returns:
            Tuple of (cluster_labels, latent_space)
        """
        self.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data, dtype=torch.float32).to(device)
            z = self.encoder(data_tensor)
            q = self.clustering_layer(z)
            
            labels = torch.argmax(q, dim=1).cpu().numpy()
            z_np = z.cpu().numpy()
        
        return labels, z_np
    
    def get_cluster_centers(self) -> np.ndarray:
        """Get current cluster centers."""
        return self.clustering_layer.cluster_centers.detach().cpu().numpy()
    
    def save_model(self, path: str) -> None:
        """
        Save complete DEC model (encoder + decoder + clustering layer).
        
        Args:
            path: Path to save the model (e.g., 'model.pth')
        """
        state = {
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'clustering_layer_state_dict': self.clustering_layer.state_dict(),
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'n_clusters': self.n_clusters,
            'pretrained': self.pretrained,
            'dec_trained': self.dec_trained,
        }
        torch.save(state, path)
        print(f"DEC model saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'DECAutoEncoder':
        """
        Load complete DEC model from file.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded DECAutoEncoder instance
        """
        state = torch.load(path, map_location=device)
        
        # Create model with saved dimensions
        model = cls(
            input_dim=state['input_dim'],
            latent_dim=state['latent_dim'],
            n_clusters=state['n_clusters']
        )
        
        # Load state dicts
        model.encoder.load_state_dict(state['encoder_state_dict'])
        model.decoder.load_state_dict(state['decoder_state_dict'])
        model.clustering_layer.load_state_dict(state['clustering_layer_state_dict'])
        
        # Restore flags
        model.pretrained = state.get('pretrained', True)
        model.dec_trained = state.get('dec_trained', True)
        
        model.eval()
        print(f"DEC model loaded from {path}")
        
        return model
