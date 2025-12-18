"""Proximal Policy Optimization (PPO) implementation."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    n_epochs: int = 4
    batch_size: int = 64
    use_amp: bool = False


class PPOTrainer:
    """PPO trainer for policy optimization."""
    
    def __init__(
        self,
        policy: nn.Module,
        config: PPOConfig,
        device: torch.device,
    ):
        self.policy = policy
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.use_amp else None
        
        # Metrics
        self.metrics = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state [1]
        
        Returns:
            advantages: GAE advantages [T]
            returns: Discounted returns [T]
        """
        T = len(rewards)
        advantages = torch.zeros(T, device=self.device)
        returns = torch.zeros(T, device=self.device)
        
        gae = 0
        next_value = next_value.item()
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1].item()
            
            delta = rewards[t] + self.config.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * gae
            
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def update(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            observations: Dict of observation tensors
            actions: Actions taken [batch]
            old_log_probs: Log probs from old policy [batch]
            advantages: GAE advantages [batch]
            returns: Discounted returns [batch]
            action_masks: Valid action masks [batch, n_actions]
        
        Returns:
            metrics: Training metrics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        epoch_metrics = {k: [] for k in self.metrics.keys()}
        
        for epoch in range(self.config.n_epochs):
            # Mini-batch training
            indices = torch.randperm(len(actions), device=self.device)
            
            for start in range(0, len(actions), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                # Get batch
                batch_obs = {k: v[batch_indices] for k, v in observations.items()}
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = action_masks[batch_indices] if action_masks is not None else None
                
                # Forward pass
                log_probs, values, entropy = self.policy.evaluate_actions(
                    signature=batch_obs["signature"],
                    tableau=batch_obs["tableau"],
                    features=batch_obs["features"],
                    actions=batch_actions,
                    action_mask=batch_masks,
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon,
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    approx_kl = (batch_old_log_probs - log_probs).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
                
                epoch_metrics["policy_loss"].append(policy_loss.item())
                epoch_metrics["value_loss"].append(value_loss.item())
                epoch_metrics["entropy"].append(-entropy_loss.item())
                epoch_metrics["total_loss"].append(loss.item())
                epoch_metrics["approx_kl"].append(approx_kl.item())
                epoch_metrics["clip_fraction"].append(clip_fraction.item())
        
        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
