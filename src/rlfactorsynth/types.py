"""Common type definitions."""

from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch


# Type aliases
Array = np.ndarray
Tensor = torch.Tensor
Device = Union[str, torch.device]

# Action types
Action = int
ActionMask = np.ndarray

# Observation types
@dataclass
class Observation:
    """RL environment observation."""
    
    signature_tensor: Array  # Factorized non-Clifford structure
    clifford_tableau: Array  # Binary symplectic representation
    structural_features: Array  # T-count, depth, etc.
    action_mask: ActionMask  # Valid actions
    
    def to_dict(self) -> Dict[str, Array]:
        """Convert to dictionary."""
        return {
            "signature_tensor": self.signature_tensor,
            "clifford_tableau": self.clifford_tableau,
            "structural_features": self.structural_features,
            "action_mask": self.action_mask,
        }


@dataclass
class StepResult:
    """Result of environment step."""
    
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


@dataclass
class Episode:
    """Complete episode data."""
    
    observations: List[Observation]
    actions: List[Action]
    rewards: List[float]
    dones: List[bool]
    infos: List[Dict[str, Any]]
    
    def __len__(self) -> int:
        return len(self.actions)


@dataclass
class Trajectory:
    """Expert trajectory for imitation learning."""
    
    target_unitary: Array
    gate_sequence: List[str]
    t_count: int
    clifford_count: int
    success: bool
