"""Reward functions and curriculum learning."""

import numpy as np
from typing import Dict, Callable


class RewardScheduler:
    """
    Dynamic reward weight scheduler for curriculum learning.
    
    Gradually shifts emphasis from finding solutions (early) to optimizing T-count (late).
    """
    
    def __init__(
        self,
        alpha_initial: float = 0.1,
        alpha_final: float = 2.0,
        beta_initial: float = 0.01,
        beta_final: float = 0.1,
        gamma_initial: float = 0.0,
        gamma_final: float = 0.05,
        total_steps: int = 500000,
        schedule_type: str = "linear",
    ):
        """
        Args:
            alpha_initial/final: T-gate cost weight (initial/final)
            beta_initial/final: Clifford cost weight (initial/final)
            gamma_initial/final: Depth penalty weight (initial/final)
            total_steps: Total training steps for curriculum
            schedule_type: 'linear', 'exponential', or 'cosine'
        """
        self.alpha_initial = alpha_initial
        self.alpha_final = alpha_final
        self.beta_initial = beta_initial
        self.beta_final = beta_final
        self.gamma_initial = gamma_initial
        self.gamma_final = gamma_final
        self.total_steps = total_steps
        self.schedule_type = schedule_type
        
        self.current_step = 0
    
    def step(self) -> None:
        """Advance one step."""
        self.current_step += 1
    
    def get_weights(self) -> Dict[str, float]:
        """Get current reward weights."""
        progress = min(1.0, self.current_step / self.total_steps)
        
        if self.schedule_type == "linear":
            alpha = self._linear_schedule(self.alpha_initial, self.alpha_final, progress)
            beta = self._linear_schedule(self.beta_initial, self.beta_final, progress)
            gamma = self._linear_schedule(self.gamma_initial, self.gamma_final, progress)
        elif self.schedule_type == "exponential":
            alpha = self._exp_schedule(self.alpha_initial, self.alpha_final, progress)
            beta = self._exp_schedule(self.beta_initial, self.beta_final, progress)
            gamma = self._exp_schedule(self.gamma_initial, self.gamma_final, progress)
        elif self.schedule_type == "cosine":
            alpha = self._cosine_schedule(self.alpha_initial, self.alpha_final, progress)
            beta = self._cosine_schedule(self.beta_initial, self.beta_final, progress)
            gamma = self._cosine_schedule(self.gamma_initial, self.gamma_final, progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        return {"alpha": alpha, "beta": beta, "gamma": gamma}
    
    @staticmethod
    def _linear_schedule(start: float, end: float, progress: float) -> float:
        """Linear interpolation."""
        return start + (end - start) * progress
    
    @staticmethod
    def _exp_schedule(start: float, end: float, progress: float) -> float:
        """Exponential schedule."""
        return start * (end / start) ** progress
    
    @staticmethod
    def _cosine_schedule(start: float, end: float, progress: float) -> float:
        """Cosine annealing schedule."""
        return end + (start - end) * 0.5 * (1 + np.cos(np.pi * progress))


class MultiObjectiveReward:
    """
    Multi-objective reward function with dynamic weighting.
    
    r = +100 (exact synthesis)
        - α(t) × T_cost(a)
        - β(t) × Clifford_cost(a)
        - γ(t) × depth_penalty(a)
    """
    
    def __init__(
        self,
        success_reward: float = 100.0,
        t_cost: float = 1.0,
        clifford_cost: float = 0.1,
        depth_cost: float = 0.01,
        scheduler: RewardScheduler = None,
    ):
        self.success_reward = success_reward
        self.t_cost = t_cost
        self.clifford_cost = clifford_cost
        self.depth_cost = depth_cost
        self.scheduler = scheduler
    
    def compute(
        self,
        is_success: bool,
        is_t_gate: bool,
        is_clifford_gate: bool,
        depth_increase: int = 1,
    ) -> float:
        """Compute reward for a step."""
        if is_success:
            return self.success_reward
        
        # Get current weights
        if self.scheduler:
            weights = self.scheduler.get_weights()
            alpha = weights["alpha"]
            beta = weights["beta"]
            gamma = weights["gamma"]
        else:
            alpha = 1.0
            beta = 1.0
            gamma = 1.0
        
        reward = 0.0
        
        # T-gate penalty
        if is_t_gate:
            reward -= alpha * self.t_cost
        
        # Clifford penalty
        if is_clifford_gate:
            reward -= beta * self.clifford_cost
        
        # Depth penalty
        reward -= gamma * self.depth_cost * depth_increase
        
        return reward


class PrecisionCurriculum:
    """
    Curriculum learning for precision: start with coarse, gradually increase precision.
    """
    
    def __init__(
        self,
        initial_epsilon: float = 1e-3,
        final_epsilon: float = 1e-6,
        total_steps: int = 500000,
    ):
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.total_steps = total_steps
        self.current_step = 0
    
    def step(self) -> None:
        """Advance one step."""
        self.current_step += 1
    
    def get_epsilon(self) -> float:
        """Get current precision threshold."""
        progress = min(1.0, self.current_step / self.total_steps)
        
        # Exponential schedule (log-linear in epsilon)
        log_initial = np.log10(self.initial_epsilon)
        log_final = np.log10(self.final_epsilon)
        log_epsilon = log_initial + (log_final - log_initial) * progress
        
        return 10 ** log_epsilon
