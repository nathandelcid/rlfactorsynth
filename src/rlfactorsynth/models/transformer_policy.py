"""Transformer-based policy and value network."""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .encoders import FactorizedStateEncoder


class TransformerPolicy(nn.Module):
    """
    Transformer policy/value network for unitary synthesis.
    
    Architecture:
    - Factorized state encoder (signature + tableau + features)
    - Transformer layers for capturing gate dependencies
    - Policy head (masked softmax over actions)
    - Value head (advantage estimation)
    """
    
    def __init__(
        self,
        signature_input_dim: int,
        tableau_input_dim: int,
        features_input_dim: int,
        n_actions: int,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_checkpointing: bool = False,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_checkpointing = use_checkpointing
        
        # State encoder
        self.encoder = FactorizedStateEncoder(
            signature_input_dim=signature_input_dim,
            tableau_input_dim=tableau_input_dim,
            features_input_dim=features_input_dim,
            fusion_dim=d_model,
            dropout=dropout,
        )
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        signature: torch.Tensor,
        tableau: torch.Tensor,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            signature: Signature tensor [batch, sig_dim]
            tableau: Clifford tableau [batch, tab_dim]
            features: Structural features [batch, feat_dim]
            action_mask: Valid action mask [batch, n_actions]
        
        Returns:
            logits: Action logits [batch, n_actions]
            value: State value [batch, 1]
        """
        # Encode state
        state = self.encoder(signature, tableau, features)  # [batch, d_model]
        
        # Add sequence dimension for transformer
        state = state.unsqueeze(1)  # [batch, 1, d_model]
        
        # Transformer processing
        if self.use_checkpointing and self.training:
            state = torch.utils.checkpoint.checkpoint(
                self.transformer, state, use_reentrant=False
            )
        else:
            state = self.transformer(state)  # [batch, 1, d_model]
        
        # Remove sequence dimension
        state = state.squeeze(1)  # [batch, d_model]
        
        # Policy head
        logits = self.policy_head(state)  # [batch, n_actions]
        
        # Apply action mask
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float("-inf"))
        
        # Value head
        value = self.value_head(state)  # [batch, 1]
        
        return logits, value
    
    def get_action_distribution(
        self,
        signature: torch.Tensor,
        tableau: torch.Tensor,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.distributions.Categorical:
        """Get action probability distribution."""
        logits, _ = self.forward(signature, tableau, features, action_mask)
        return torch.distributions.Categorical(logits=logits)
    
    def act(
        self,
        signature: torch.Tensor,
        tableau: torch.Tensor,
        features: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action [batch]
            log_prob: Log probability of action [batch]
            value: State value [batch]
        """
        logits, value = self.forward(signature, tableau, features, action_mask)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        signature: torch.Tensor,
        tableau: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO updates.
        
        Returns:
            log_probs: Log probabilities of actions [batch]
            values: State values [batch]
            entropy: Policy entropy [batch]
        """
        logits, values = self.forward(signature, tableau, features, action_mask)
        
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
