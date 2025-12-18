#!/usr/bin/env python3
"""Train RLFactorSynth policy using PPO."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlfactorsynth.envs.synthesis_env import SynthesisEnv, SynthesisConfig
from rlfactorsynth.envs.rewards import RewardScheduler
from rlfactorsynth.models.transformer_policy import TransformerPolicy
from rlfactorsynth.rl.ppo import PPOTrainer, PPOConfig
from rlfactorsynth.quantum.unitary import random_unitary
from rlfactorsynth.utils.seeding import set_seed
from rlfactorsynth.utils.device import get_device


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main training loop."""
    
    print("=" * 80)
    print("RLFactorSynth PPO Training")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    set_seed(cfg.experiment.seed, cfg.experiment.deterministic)
    
    # Device
    device = get_device(cfg.device.type)
    print(f"\nUsing device: {device}")
    
    # Create environment
    env_config = SynthesisConfig(
        n_qubits=cfg.env.n_qubits,
        max_steps=cfg.env.max_steps,
        success_threshold=cfg.env.success_threshold,
        primitive_gates=cfg.env.gates.primitives,
        use_meta_actions=cfg.env.gates.use_meta_actions,
    )
    env = SynthesisEnv(env_config)
    
    print(f"\nEnvironment: {cfg.env.name}")
    print(f"  n_qubits: {env.n_qubits}")
    print(f"  n_actions: {env.n_actions}")
    print(f"  max_steps: {env.config.max_steps}")
    
    # Compute observation dimensions
    dummy_unitary = random_unitary(env.n_qubits, seed=42)
    dummy_obs = env.reset(dummy_unitary)
    
    sig_dim = len(dummy_obs.signature_tensor)
    tab_dim = len(dummy_obs.clifford_tableau)
    feat_dim = len(dummy_obs.structural_features)
    
    print(f"\nObservation dimensions:")
    print(f"  Signature: {sig_dim}")
    print(f"  Tableau: {tab_dim}")
    print(f"  Features: {feat_dim}")
    
    # Create policy
    policy = TransformerPolicy(
        signature_input_dim=sig_dim,
        tableau_input_dim=tab_dim,
        features_input_dim=feat_dim,
        n_actions=env.n_actions,
        d_model=cfg.model.transformer.d_model,
        nhead=cfg.model.transformer.nhead,
        num_layers=cfg.model.transformer.num_layers,
        dim_feedforward=cfg.model.transformer.dim_feedforward,
        dropout=cfg.model.transformer.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"\nPolicy parameters: {n_params:,}")
    
    # Load checkpoint if specified
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        print(f"\nLoading checkpoint: {cfg.checkpoint_path}")
        policy.load_state_dict(torch.load(cfg.checkpoint_path, map_location=device))
    
    # Create PPO trainer
    ppo_config = PPOConfig(
        learning_rate=cfg.train.learning_rate,
        gamma=cfg.train.gamma,
        gae_lambda=cfg.train.gae_lambda,
        clip_epsilon=cfg.train.clip_epsilon,
        n_epochs=cfg.train.n_epochs,
        batch_size=cfg.train.batch_size,
    )
    trainer = PPOTrainer(policy, ppo_config, device)
    
    # Reward scheduler
    reward_scheduler = RewardScheduler(
        alpha_initial=cfg.env.reward.alpha_initial,
        alpha_final=cfg.env.reward.alpha_final,
        beta_initial=cfg.env.reward.beta_initial,
        beta_final=cfg.env.reward.beta_final,
        total_steps=cfg.train.total_steps,
    )
    
    # Training loop
    print(f"\n{'=' * 80}")
    print("Starting training...")
    print(f"{'=' * 80}\n")
    
    total_steps = 0
    episode_rewards = []
    episode_successes = []
    
    checkpoint_dir = Path(cfg.experiment.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    pbar = tqdm(total=cfg.train.total_steps, desc="Training")
    
    while total_steps < cfg.train.total_steps:
        # Generate random target unitary
        target = random_unitary(env.n_qubits)
        obs = env.reset(target)
        
        # Collect episode
        episode_data = {
            "observations": {"signature": [], "tableau": [], "features": []},
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": [],
            "dones": [],
            "action_masks": [],
        }
        
        done = False
        episode_reward = 0.0
        
        while not done and total_steps < cfg.train.total_steps:
            # Convert observation to tensors
            sig_tensor = torch.FloatTensor(obs.signature_tensor).unsqueeze(0).to(device)
            tab_tensor = torch.FloatTensor(obs.clifford_tableau).unsqueeze(0).to(device)
            feat_tensor = torch.FloatTensor(obs.structural_features).unsqueeze(0).to(device)
            mask_tensor = torch.BoolTensor(obs.action_mask).unsqueeze(0).to(device)
            
            # Sample action
            with torch.no_grad():
                action, log_prob, value = policy.act(
                    sig_tensor, tab_tensor, feat_tensor, mask_tensor
                )
            
            # Step environment
            result = env.step(action.item())
            
            # Store transition
            episode_data["observations"]["signature"].append(obs.signature_tensor)
            episode_data["observations"]["tableau"].append(obs.clifford_tableau)
            episode_data["observations"]["features"].append(obs.structural_features)
            episode_data["actions"].append(action.item())
            episode_data["log_probs"].append(log_prob.item())
            episode_data["rewards"].append(result.reward)
            episode_data["values"].append(value.item())
            episode_data["dones"].append(result.done)
            episode_data["action_masks"].append(obs.action_mask)
            
            episode_reward += result.reward
            obs = result.observation
            done = result.done
            
            total_steps += 1
            reward_scheduler.step()
            pbar.update(1)
        
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_successes.append(result.info["success"])
        
        # PPO update (simplified - normally would accumulate multiple episodes)
        if len(episode_data["actions"]) > 0:
            # Convert to tensors
            obs_dict = {
                "signature": torch.FloatTensor(np.array(episode_data["observations"]["signature"])).to(device),
                "tableau": torch.FloatTensor(np.array(episode_data["observations"]["tableau"])).to(device),
                "features": torch.FloatTensor(np.array(episode_data["observations"]["features"])).to(device),
            }
            actions = torch.LongTensor(episode_data["actions"]).to(device)
            old_log_probs = torch.FloatTensor(episode_data["log_probs"]).to(device)
            rewards = torch.FloatTensor(episode_data["rewards"]).to(device)
            values = torch.FloatTensor(episode_data["values"]).to(device)
            dones = torch.FloatTensor(episode_data["dones"]).to(device)
            
            # Compute GAE
            advantages, returns = trainer.compute_gae(
                rewards, values, dones, torch.tensor([0.0])
            )
            
            # Update policy
            metrics = trainer.update(
                obs_dict, actions, old_log_probs, advantages, returns
            )
        
        # Logging
        if len(episode_rewards) % cfg.logging.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            success_rate = np.mean(episode_successes[-100:])
            pbar.set_postfix({
                "reward": f"{avg_reward:.2f}",
                "success": f"{success_rate:.2%}",
            })
        
        # Save checkpoint
        if total_steps % cfg.logging.save_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
            torch.save(policy.state_dict(), checkpoint_path)
    
    pbar.close()
    
    # Save final model
    final_path = checkpoint_dir / "final.pt"
    torch.save(policy.state_dict(), final_path)
    print(f"\nSaved final model to: {final_path}")
    
    print(f"\n{'=' * 80}")
    print("Training complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
