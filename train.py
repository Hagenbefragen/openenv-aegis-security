"""
AEGIS RL Training Script — Automated Stellschrauben Optimization
OpenEnv Hackathon SF (March 2026)

Uses φ-damped evolutionary strategy to find optimal detection thresholds.
No PyTorch needed — pure Python hill-climbing with bounded evolution.

How it works:
  1. Start with default V41 Stellschrauben (42 thresholds)
  2. Each epoch: randomly perturb thresholds within SIREN bounds
  3. Evaluate: run full corpus through cascade
  4. If reward improved → keep changes (φ-damped momentum)
  5. If reward decreased → revert (bounded: can't go worse)
  6. Fibonacci checkpoints save best configuration

This is the SIREN Evolution Engine pattern adapted for RL:
  - Hard bounds prevent threshold collapse
  - φ-damping prevents oscillation
  - Monotonic improvement guarantee (bounded evolution)
"""

import random
import json
import time
import math
from aegis_env import AEGISSecurityEnv, AEGISAction

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]


def train(
    epochs: int = 50,
    steps_per_episode: int = 40,
    perturbation_scale: float = 0.08,
    phi_decay: float = PHI_INV,
    verbose: bool = True,
):
    """Train AEGIS cascade using evolutionary strategy.
    
    Args:
        epochs: Number of training epochs
        steps_per_episode: Prompts per episode
        perturbation_scale: How much to perturb thresholds each epoch
        phi_decay: φ⁻¹ learning rate decay
        verbose: Print training progress
    """
    env = AEGISSecurityEnv(max_steps=steps_per_episode)
    
    # Initial baseline episode
    best_reward = _run_episode(env)
    best_stellschrauben = dict(env.cascade.get_stellschrauben())
    best_metrics = _get_metrics(env)
    
    if verbose:
        print("=" * 70)
        print("🛡️  AEGIS RL Training — Stellschrauben Optimization")
        print("=" * 70)
        print(f"  Epochs:        {epochs}")
        print(f"  Steps/Episode: {steps_per_episode}")
        print(f"  Initial Catch: {best_metrics['catch_rate']:.1%}")
        print(f"  Initial FPR:   {best_metrics['fpr']:.1%}")
        print(f"  Initial Reward: {best_reward:.3f}")
        print(f"  Stellschrauben: {len(best_stellschrauben)} tunable thresholds")
        print("-" * 70)
    
    history = []
    improvements = 0
    scale = perturbation_scale
    
    for epoch in range(1, epochs + 1):
        # Save current state
        current_stellschrauben = dict(env.cascade.get_stellschrauben())
        
        # Perturb: randomly adjust a subset of Stellschrauben
        # Select 3-8 thresholds to adjust (not all at once — focused exploration)
        all_names = list(current_stellschrauben.keys())
        num_to_adjust = min(len(all_names), random.randint(3, 8))
        selected = random.sample(all_names, num_to_adjust)
        
        perturbations = {}
        for name in selected:
            # φ-damped random perturbation
            delta = random.gauss(0, scale) * phi_decay
            new_val = current_stellschrauben[name] + delta
            # Hard bounds: 0.05 ≤ threshold ≤ 0.95 (SIREN safety rails)
            new_val = max(0.05, min(0.95, new_val))
            env.cascade.set_stellschraube(name, new_val)
            perturbations[name] = (current_stellschrauben[name], new_val)
        
        # Evaluate with new thresholds
        episode_reward = _run_episode(env)
        episode_metrics = _get_metrics(env)
        
        # Decision: keep or revert?
        if episode_reward > best_reward:
            # ✅ IMPROVEMENT — keep the changes
            best_reward = episode_reward
            best_stellschrauben = dict(env.cascade.get_stellschrauben())
            best_metrics = episode_metrics
            improvements += 1
            status = "✅ IMPROVED"
            
            # Reduce exploration (we're getting closer to optimal)
            scale *= 0.98
        else:
            # ❌ WORSE — revert to best known configuration
            for name, val in best_stellschrauben.items():
                env.cascade.set_stellschraube(name, val)
            status = "↩️  reverted"
            
            # Increase exploration (we need to try harder)
            scale = min(perturbation_scale, scale * 1.02)
        
        # Log history
        entry = {
            "epoch": epoch,
            "reward": episode_reward,
            "best_reward": best_reward,
            "catch_rate": episode_metrics["catch_rate"],
            "fpr": episode_metrics["fpr"],
            "accuracy": episode_metrics["accuracy"],
            "latency_ms": episode_metrics["latency_ms"],
            "scale": scale,
            "status": status,
        }
        history.append(entry)
        
        # Print progress
        if verbose:
            is_fib = epoch in FIBONACCI
            prefix = "🌀" if is_fib else "  "
            print(f"{prefix} Epoch {epoch:3d} | "
                  f"catch={episode_metrics['catch_rate']:.1%} "
                  f"FPR={episode_metrics['fpr']:.1%} "
                  f"reward={episode_reward:7.3f} "
                  f"best={best_reward:7.3f} | "
                  f"{status}")
            
            if is_fib:
                # Fibonacci checkpoint — save best config
                checkpoint = {
                    "epoch": epoch,
                    "stellschrauben": best_stellschrauben,
                    "metrics": best_metrics,
                    "reward": best_reward,
                }
                fname = f"checkpoint_epoch_{epoch}.json"
                with open(fname, "w") as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"     💾 Saved checkpoint: {fname}")
    
    # Final report
    if verbose:
        print()
        print("=" * 70)
        print("🏆 TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Total Epochs:    {epochs}")
        print(f"  Improvements:    {improvements}/{epochs} ({improvements/epochs:.0%})")
        print(f"  Final Catch Rate: {best_metrics['catch_rate']:.1%}")
        print(f"  Final FPR:        {best_metrics['fpr']:.1%}")
        print(f"  Final Accuracy:   {best_metrics['accuracy']:.1%}")
        print(f"  Final Latency:    {best_metrics['latency_ms']:.1f}ms")
        print(f"  Final Reward:     {best_reward:.3f}")
        print()
        
        # Show top Stellschrauben changes
        env_default = AEGISSecurityEnv(max_steps=1)
        default_s = env_default.cascade.get_stellschrauben()
        
        print("  📊 Stellschrauben Changes (Top 10):")
        changes = []
        for name in best_stellschrauben:
            if name in default_s:
                delta = best_stellschrauben[name] - default_s[name]
                if abs(delta) > 0.001:
                    changes.append((name, default_s[name], best_stellschrauben[name], delta))
        
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        for name, old, new, delta in changes[:10]:
            arrow = "↑" if delta > 0 else "↓"
            print(f"     {arrow} {name}: {old:.3f} → {new:.3f} ({delta:+.3f})")
    
    # Save final config
    final = {
        "stellschrauben": best_stellschrauben,
        "metrics": best_metrics,
        "reward": best_reward,
        "epochs": epochs,
        "improvements": improvements,
        "history": history,
    }
    with open("trained_stellschrauben.json", "w") as f:
        json.dump(final, f, indent=2)
    print(f"\n  💾 Saved: trained_stellschrauben.json")
    
    return final


def _run_episode(env: AEGISSecurityEnv) -> float:
    """Run one full episode and return total reward."""
    state = env.reset()
    total_reward = 0.0
    
    while not state.done:
        pair = env.get_next_prompt()
        if not pair:
            break
        prompt, expected = pair
        obs = env.step(AEGISAction(prompt=prompt, expected_category=expected))
        total_reward += obs.reward
        state = env.state()
    
    return total_reward


def _get_metrics(env: AEGISSecurityEnv) -> dict:
    """Get current episode metrics as dict."""
    state = env.state()
    return {
        "catch_rate": state.metrics.catch_rate,
        "fpr": state.metrics.fpr,
        "accuracy": state.metrics.accuracy,
        "latency_ms": state.metrics.avg_latency_ms,
    }


if __name__ == "__main__":
    import sys
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    result = train(epochs=epochs)
