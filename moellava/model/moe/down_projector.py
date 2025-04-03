import torch
from torch import nn
from typing import Tuple

class Down_Projector_v1(nn.Module):
    def __init__(
        self,
        model_dim: int,         # D: dimension of input features
        projection_dim: int = 4,  # New parameter for reduced dimension
    ):
        super().__init__()
        self.model_dim = model_dim
        self.projection_dim = projection_dim
        # Dimension reduction layers
        self.down_proj_encoder = nn.Sequential(
            nn.Linear(model_dim, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, projection_dim)
        )
        self.down_proj_decoder = nn.Sequential(
            nn.Linear(projection_dim, model_dim // 4),
            nn.ReLU(),
            nn.Linear(model_dim // 4, model_dim)
        )
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        input,
    ):
        low_dim_embedding = self.down_proj_encoder(input)
        high_dim_recover = self.down_proj_decoder(low_dim_embedding)
        recover_loss = self.loss_fn(high_dim_recover,input)
        return low_dim_embedding, recover_loss


class Down_Projector(nn.Module):
    def __init__(
        self,
        model_dim: int,
        projection_dim: int = 4,
        hidden_factor: int = 4,  # Configurable reduction factor for hidden layer
        activation: nn.Module = nn.GELU(),  # Better activation function
        layer_norm: bool = True,  # Add layer normalization
        dropout: float = 0.1,  # Add dropout for regularization
    ):
        super().__init__()
        self.model_dim = model_dim
        self.projection_dim = projection_dim
        
        hidden_dim = model_dim // hidden_factor
        
        # Enhanced encoder with LayerNorm and Dropout
        encoder_layers = [
            nn.Linear(model_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        ]
        if layer_norm:
            encoder_layers.insert(0, nn.LayerNorm(model_dim))
            encoder_layers.insert(-1, nn.LayerNorm(hidden_dim))
        self.down_proj_encoder = nn.Sequential(*encoder_layers)
        
        # Enhanced decoder with LayerNorm and Dropout
        decoder_layers = [
            nn.Linear(projection_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, model_dim),
        ]
        if layer_norm:
            decoder_layers.insert(0, nn.LayerNorm(projection_dim))
            decoder_layers.insert(-1, nn.LayerNorm(hidden_dim))
        self.down_proj_decoder = nn.Sequential(*decoder_layers)
        
        self.loss_fn = nn.MSELoss()

    def forward(
        self,
        input: torch.Tensor,
        return_reconstruction: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the projector
        
        Args:
            input: Input tensor of shape (batch_size, model_dim)
            return_reconstruction: If True, also return the reconstructed input
            
        Returns:
            Tuple containing:
            - low_dim_embedding: Projected tensor of shape (batch_size, projection_dim)
            - recover_loss: Reconstruction loss
            - high_dim_recover: (optional) Reconstructed input tensor
        """
        low_dim_embedding = self.down_proj_encoder(input)
        high_dim_recover = self.down_proj_decoder(low_dim_embedding)
        recover_loss = self.loss_fn(high_dim_recover, input)
        
        if return_reconstruction:
            return low_dim_embedding, recover_loss, high_dim_recover
        return low_dim_embedding, recover_loss

    @torch.no_grad()
    def project(self, input: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for inference-only projection
        """
        return self.down_proj_encoder(input)