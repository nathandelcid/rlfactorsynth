#!/usr/bin/env python3
"""Synthesize unitaries using trained RLFactorSynth policy."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
import argparse
from pathlib import Path
import time

from rlfactorsynth.envs.synthesis_env import SynthesisEnv, SynthesisConfig
from rlfactorsynth.models.transformer_policy import TransformerPolicy
from rlfactorsynth.quantum.unitary import random_unitary, unitary_distance
from rlfactorsynth.utils.device import get_device


def load_policy(checkpoint_path: str, env: SynthesisEnv, device: torch.device):
    """Load trained policy from checkpoint."""
    # Get observation dimensions
    dummy_unitary = random_unitary(env.n_qubits, seed=42)
    dummy_obs = env.reset(dummy_unitary)
    
    sig_dim = len(dummy_obs.signature_tensor)
    tab_dim = len(dummy_obs.clifford_tableau)
    feat_dim = len(dummy_obs.structural_features)
    
    # Create policy
    policy = TransformerPolicy(
        signature_input_dim=sig_dim,
        tableau_input_dim=tab_dim,
        features_input_dim=feat_dim,
        n_actions=env.n_actions,
    ).to(device)
    
    # Load weights
    policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
    policy.eval()
    
    return policy


def synthesize_unitary(
    target: np.ndarray,
    policy: torch.nn.Module,
    env: SynthesisEnv,
    device: torch.device,
    deterministic: bool = True,
) -> dict:
    """
    Synthesize a single unitary.
    
    Returns:
        result: Dict with gate_sequence, t_count, success, etc.
    """
    start_time = time.time()
    
    obs = env.reset(target)
    done = False
    
    while not done:
        # Convert observation to tensors
        sig_tensor = torch.FloatTensor(obs.signature_tensor).unsqueeze(0).to(device)
        tab_tensor = torch.FloatTensor(obs.clifford_tableau).unsqueeze(0).to(device)
        feat_tensor = torch.FloatTensor(obs.structural_features).unsqueeze(0).to(device)
        mask_tensor = torch.BoolTensor(obs.action_mask).unsqueeze(0).to(device)
        
        # Get action
        with torch.no_grad():
            action, _, _ = policy.act(
                sig_tensor, tab_tensor, feat_tensor, mask_tensor,
                deterministic=deterministic
            )
        
        # Step
        result = env.step(action.item())
        obs = result.observation
        done = result.done
    
    synthesis_time = time.time() - start_time
    
    return {
        "gate_sequence": env.gate_history,
        "t_count": env.t_count,
        "clifford_count": env.clifford_count,
        "total_gates": len(env.gate_history),
        "success": result.info["success"],
        "final_distance": result.info["distance"],
        "synthesis_time": synthesis_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthesize unitaries with RLFactorSynth")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--unitary_path", type=str, help="Path to target unitary (.npy)")
    parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits (for random)")
    parser.add_argument("--n_random", type=int, default=1, help="Number of random unitaries")
    parser.add_argument("--mode", type=str, default="balanced", 
                       choices=["ultra_fast", "balanced", "quality"],
                       help="Operating mode")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RLFactorSynth Synthesis")
    print("=" * 80)
    
    # Device
    device = get_device(args.device)
    print(f"Device: {device}")
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create environment
    env_config = SynthesisConfig(
        n_qubits=args.n_qubits,
        max_steps=100,
    )
    env = SynthesisEnv(env_config)
    
    print(f"Environment: {args.n_qubits}-qubit synthesis")
    print(f"Actions: {env.n_actions}")
    
    # Load policy
    print(f"\nLoading checkpoint: {args.checkpoint}")
    policy = load_policy(args.checkpoint, env, device)
    print("Policy loaded successfully")
    
    # Load or generate unitaries
    unitaries = []
    
    if args.unitary_path:
        print(f"\nLoading unitary from: {args.unitary_path}")
        target = np.load(args.unitary_path)
        unitaries.append(target)
    else:
        print(f"\nGenerating {args.n_random} random unitaries...")
        for i in range(args.n_random):
            target = random_unitary(args.n_qubits, seed=args.seed + i)
            unitaries.append(target)
    
    # Synthesize
    print(f"\n{'=' * 80}")
    print("Synthesizing...")
    print(f"{'=' * 80}\n")
    
    results = []
    
    for i, target in enumerate(unitaries):
        print(f"Unitary {i+1}/{len(unitaries)}")
        
        result = synthesize_unitary(target, policy, env, device)
        results.append(result)
        
        print(f"  Success: {result['success']}")
        print(f"  T-count: {result['t_count']}")
        print(f"  Total gates: {result['total_gates']}")
        print(f"  Time: {result['synthesis_time']:.3f}s")
        print(f"  Final distance: {result['final_distance']:.2e}")
        print()
    
    # Summary
    print(f"{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_t_count = np.mean([r["t_count"] for r in results])
    avg_time = np.mean([r["synthesis_time"] for r in results])
    
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average T-count: {avg_t_count:.1f}")
    print(f"Average synthesis time: {avg_time:.3f}s")
    
    # Throughput
    if len(results) > 1:
        total_time = sum(r["synthesis_time"] for r in results)
        throughput = len(results) / total_time
        print(f"Throughput: {throughput:.2f} unitaries/second")


if __name__ == "__main__":
    main()
