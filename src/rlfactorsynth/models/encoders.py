"""Neural network encoders for factorized state representation."""

import torch
import torch.nn as nn
from typing import Tuple


class SignatureTensorEncoder(nn.Module):
    """
    Encoder for signature tensor (low-rank factorization of residual unitary).
    
    Processes the factorized representation of non-Clifford structure.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Signature tensor features [batch, input_dim]
        
        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)


class CliffordTableauEncoder(nn.Module):
    """
    Encoder for Clifford tableau (binary symplectic representation).
    
    Processes the efficient Clifford tracking state.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Clifford tableau [batch, input_dim]
        
        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)


class StructuralFeaturesEncoder(nn.Module):
    """
    Encoder for structural features (T-count, depth, gate history, etc.).
    """
    
    def __init__(
        self,
        input_dim: int = 16,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Structural features [batch, input_dim]
        
        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)


class FactorizedStateEncoder(nn.Module):
    """
    Complete factorized state encoder combining all components.
    
    Encodes: signature tensor + Clifford tableau + structural features → fused representation
    """
    
    def __init__(
        self,
        signature_input_dim: int,
        tableau_input_dim: int,
        features_input_dim: int = 16,
        signature_output_dim: int = 512,
        tableau_output_dim: int = 256,
        features_output_dim: int = 128,
        fusion_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Individual encoders
        self.signature_encoder = SignatureTensorEncoder(
            signature_input_dim,
            hidden_dim=256,
            output_dim=signature_output_dim,
            num_layers=2,
            dropout=dropout,
        )
        
        self.tableau_encoder = CliffordTableauEncoder(
            tableau_input_dim,
            hidden_dim=128,
            output_dim=tableau_output_dim,
            num_layers=2,
            dropout=dropout,
        )
        
        self.features_encoder = StructuralFeaturesEncoder(
            features_input_dim,
            hidden_dim=64,
            output_dim=features_output_dim,
            dropout=dropout,
        )
        
        # Fusion layer
        total_dim = signature_output_dim + tableau_output_dim + features_output_dim
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        signature: torch.Tensor,
        tableau: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            signature: Signature tensor [batch, sig_dim]
            tableau: Clifford tableau [batch, tab_dim]
            features: Structural features [batch, feat_dim]
        
        Returns:
            Fused state representation [batch, fusion_dim]
        """
        # Encode each component
        sig_encoded = self.signature_encoder(signature)
        tab_encoded = self.tableau_encoder(tableau)
        feat_encoded = self.features_encoder(features)
        
        # Concatenate and fuse
        concatenated = torch.cat([sig_encoded, tab_encoded, feat_encoded], dim=-1)
        fused = self.fusion(concatenated)
        
        return fused


class RawMatrixEncoder(nn.Module):
    """
    Baseline encoder using raw flattened unitary matrix.
    
    For ablation studies comparing factorized vs. raw encoding.
    """
    
    def __init__(
        self,
        matrix_dim: int,
        hidden_dim: int = 1024,
        output_dim: int = 768,
        num_layers: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Flatten complex matrix to real vector (2 * d^2)
        input_dim = 2 * matrix_dim * matrix_dim
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Flattened unitary matrix [batch, 2*d^2]
        
        Returns:
            Encoded features [batch, output_dim]
        """
        return self.encoder(x)
