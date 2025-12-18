"""Unitary synthesis RL environment with factorized state encoding."""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..quantum.gates import GateSet, count_t_gates, count_clifford_gates
from ..quantum.unitary import (
    unitary_distance,
    compute_residual,
    pauli_transfer_matrix,
    low_rank_approximation,
)
from ..quantum.clifford_tableau import CliffordTableau
from ..types import Observation, StepResult


@dataclass
class SynthesisConfig:
    """Configuration for synthesis environment."""
    
    n_qubits: int = 1
    max_steps: int = 100
    success_threshold: float = 1e-8
    distance_metric: str = "operator"
    
    # Gate set
    primitive_gates: List[str] = None
    use_meta_actions: bool = True
    meta_action_max_length: int = 5
    
    # Reward
    success_reward: float = 100.0
    t_cost: float = 1.0
    clifford_cost: float = 0.1
    depth_cost: float = 0.01
    
    # Encoding
    signature_rank: int = 10  # Low-rank approximation rank
    
    def __post_init__(self):
        if self.primitive_gates is None:
            self.primitive_gates = ["H", "T", "Tdg", "S", "Sdg", "Z"]


class SynthesisEnv:
    """
    RL environment for unitary synthesis with factorized state encoding.
    
    State space: Factorized representation (signature tensor + Clifford tableau + features)
    Action space: Primitive gates + meta-actions
    """
    
    def __init__(self, config: SynthesisConfig):
        self.config = config
        self.n_qubits = config.n_qubits
        self.dim = 2 ** self.n_qubits
        
        # Action space
        self.actions = self._build_action_space()
        self.n_actions = len(self.actions)
        
        # State
        self.target_unitary: Optional[np.ndarray] = None
        self.current_unitary: Optional[np.ndarray] = None
        self.clifford_tableau: Optional[CliffordTableau] = None
        self.gate_history: List[str] = []
        self.step_count: int = 0
        
        # Metrics
        self.t_count: int = 0
        self.clifford_count: int = 0
    
    def _build_action_space(self) -> List[str]:
        """Build action space from primitive gates and meta-actions."""
        actions = []
        
        # Primitive gates
        for gate_name in self.config.primitive_gates:
            for qubit in range(self.n_qubits):
                actions.append(f"{gate_name}_{qubit}")
        
        # Meta-actions (placeholder - would be loaded from library)
        if self.config.use_meta_actions:
            # Example meta-actions
            meta_actions = [
                "RZ_PI8_0",  # Optimal Rz(π/8) decomposition
                "CLIFFORD_T_PATTERN_0",  # Common Clifford+T pattern
            ]
            actions.extend(meta_actions)
        
        return actions
    
    def reset(self, target_unitary: np.ndarray) -> Observation:
        """Reset environment with a new target unitary."""
        assert target_unitary.shape == (self.dim, self.dim)
        
        self.target_unitary = target_unitary
        self.current_unitary = np.eye(self.dim, dtype=np.complex128)
        self.clifford_tableau = CliffordTableau(self.n_qubits)
        self.gate_history = []
        self.step_count = 0
        self.t_count = 0
        self.clifford_count = 0
        
        return self._get_observation()
    
    def step(self, action: int) -> StepResult:
        """Execute one step in the environment."""
        # Apply action
        gate_name = self.actions[action]
        self._apply_action(gate_name)
        
        # Compute reward
        distance = self._compute_distance()
        done = distance <= self.config.success_threshold or self.step_count >= self.config.max_steps
        reward = self._compute_reward(gate_name, distance, done)
        
        # Get observation
        obs = self._get_observation()
        
        # Info
        info = {
            "distance": distance,
            "t_count": self.t_count,
            "clifford_count": self.clifford_count,
            "steps": self.step_count,
            "success": distance <= self.config.success_threshold,
        }
        
        return StepResult(obs, reward, done, info)
    
    def _apply_action(self, gate_name: str) -> None:
        """Apply a gate to the current circuit."""
        self.step_count += 1
        self.gate_history.append(gate_name)
        
        # Parse gate name
        if "_" in gate_name:
            parts = gate_name.split("_")
            gate_type = parts[0]
            qubit = int(parts[1]) if len(parts) > 1 else 0
        else:
            gate_type = gate_name
            qubit = 0
        
        # Update counts
        if gate_type in ["T", "Tdg"]:
            self.t_count += 1
        else:
            self.clifford_count += 1
        
        # Apply to unitary
        gate = GateSet.get_gate(gate_type, (qubit,))
        self.current_unitary = GateSet.apply_gate(
            self.current_unitary, gate, self.n_qubits
        )
        
        # Update Clifford tableau (if Clifford gate)
        if gate.is_clifford:
            self.clifford_tableau.apply_gate(gate_type, qubit)
    
    def _compute_distance(self) -> float:
        """Compute distance between target and current unitary."""
        return unitary_distance(
            self.target_unitary,
            self.current_unitary,
            metric=self.config.distance_metric,
        )
    
    def _compute_reward(self, gate_name: str, distance: float, done: bool) -> float:
        """Compute reward for the current step."""
        # Success reward
        if done and distance <= self.config.success_threshold:
            return self.config.success_reward
        
        # Step penalties
        reward = 0.0
        
        # T-gate penalty
        if "T" in gate_name:
            reward -= self.config.t_cost
        
        # Clifford penalty
        else:
            reward -= self.config.clifford_cost
        
        # Depth penalty
        reward -= self.config.depth_cost
        
        return reward
    
    def _get_observation(self) -> Observation:
        """Get current observation with factorized encoding."""
        # Compute residual
        residual = compute_residual(self.target_unitary, self.current_unitary)
        
        # 1. Signature tensor (low-rank factorization of residual)
        signature_tensor = self._encode_signature_tensor(residual)
        
        # 2. Clifford tableau (binary symplectic)
        clifford_tableau = self.clifford_tableau.to_binary_vector()
        
        # 3. Structural features
        structural_features = self._encode_structural_features()
        
        # 4. Action mask
        action_mask = self._compute_action_mask()
        
        return Observation(
            signature_tensor=signature_tensor,
            clifford_tableau=clifford_tableau,
            structural_features=structural_features,
            action_mask=action_mask,
        )
    
    def _encode_signature_tensor(self, residual: np.ndarray) -> np.ndarray:
        """
        Encode residual unitary as signature tensor (low-rank approximation).
        
        Returns flattened representation of low-rank factors.
        """
        rank = min(self.config.signature_rank, self.dim)
        
        # Low-rank SVD
        U, s, Vh = low_rank_approximation(residual, rank)
        
        # Flatten and concatenate (real and imaginary parts)
        features = []
        features.append(np.real(U).flatten())
        features.append(np.imag(U).flatten())
        features.append(s)
        features.append(np.real(Vh).flatten())
        features.append(np.imag(Vh).flatten())
        
        return np.concatenate(features)
    
    def _encode_structural_features(self) -> np.ndarray:
        """Encode structural features (T-count, depth, etc.)."""
        features = [
            self.t_count,
            self.clifford_count,
            self.step_count,
            len(self.gate_history),
            self._compute_distance(),
        ]
        
        # Add recent gate history (one-hot encoded)
        # Simplified: just count recent gate types
        recent_gates = self.gate_history[-10:] if len(self.gate_history) >= 10 else self.gate_history
        for gate_type in ["H", "T", "Tdg", "S", "Sdg", "Z"]:
            count = sum(1 for g in recent_gates if g.startswith(gate_type))
            features.append(count)
        
        return np.array(features, dtype=np.float32)
    
    def _compute_action_mask(self) -> np.ndarray:
        """Compute valid action mask."""
        mask = np.ones(self.n_actions, dtype=bool)
        
        # Simple masking: disable actions that would exceed max steps
        if self.step_count >= self.config.max_steps - 1:
            mask[:] = False
        
        return mask
