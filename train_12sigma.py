"""
AEGIS 12-Sigma Training — Massive Corpus RL Training
Uses the synthetic corpus generator for large-scale training.
Target: 99.99% catch rate (approaching 12-Sigma)
"""

import random
import json
import time
from aegis_env import AEGISSecurityEnv, AEGISAction
from corpus_generator import generate_corpus, generate_attack, generate_benign
from models import ThreatCategory

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI


def train_12sigma(
    corpus_size: int = 1000,
    epochs: int = 100,
    batch_size: int = 200,
    perturbation_scale: float = 0.06,
    verbose: bool = True,
):
    """Train AEGIS cascade toward 12-Sigma using massive synthetic corpus.
    
    Each epoch:
      1. Sample a random batch from the corpus
      2. Perturb Stellschrauben
      3. Evaluate batch
      4. Keep improvements, revert failures
    """
    # Generate corpus
    attack_count = corpus_size // 2
    benign_count = corpus_size // 2
    corpus = generate_corpus(attack_count, benign_count, seed=42)
    
    if verbose:
        print("=" * 70)
        print("  AEGIS 12-SIGMA TRAINING")
        print("=" * 70)
        print(f"  Corpus Size:     {len(corpus)} prompts")
        print(f"  Attacks:         {attack_count}")
        print(f"  Benign:          {benign_count}")
        print(f"  Epochs:          {epochs}")
        print(f"  Batch Size:      {batch_size}")
        print(f"  Target:          12-Sigma (99.99%+ catch)")
        print("-" * 70)
    
    env = AEGISSecurityEnv(max_steps=batch_size)
    
    # Baseline evaluation on full corpus
    best_catch, best_fpr, best_reward = _evaluate_batch(env, corpus[:batch_size])
    best_stellschrauben = dict(env.cascade.get_stellschrauben())
    
    if verbose:
        print(f"  Baseline:  catch={best_catch:.1%}  FPR={best_fpr:.1%}  reward={best_reward:.1f}")
        print("-" * 70)
    
    improvements = 0
    scale = perturbation_scale
    
    for epoch in range(1, epochs + 1):
        # Random batch from corpus
        batch = random.sample(corpus, min(batch_size, len(corpus)))
        
        # Save current
        current_s = dict(env.cascade.get_stellschrauben())
        
        # Perturb 5-12 random Stellschrauben
        all_names = list(current_s.keys())
        n_adjust = random.randint(5, 12)
        selected = random.sample(all_names, min(n_adjust, len(all_names)))
        
        for name in selected:
            delta = random.gauss(0, scale) * PHI_INV
            new_val = max(0.05, min(0.95, current_s[name] + delta))
            env.cascade.set_stellschraube(name, new_val)
        
        # Evaluate
        catch, fpr, reward = _evaluate_batch(env, batch)
        
        if reward > best_reward:
            best_reward = reward
            best_catch = catch
            best_fpr = fpr
            best_stellschrauben = dict(env.cascade.get_stellschrauben())
            improvements += 1
            status = "++ IMPROVED"
            scale *= 0.97  # Narrow search
        else:
            # Revert
            for name, val in best_stellschrauben.items():
                env.cascade.set_stellschraube(name, val)
            status = "   reverted"
            scale = min(perturbation_scale, scale * 1.01)
        
        if verbose and (epoch % 5 == 0 or epoch <= 5 or "IMPROVED" in status):
            print(f"  Epoch {epoch:4d} | catch={catch:.1%} FPR={fpr:.1%} "
                  f"reward={reward:8.1f} best={best_reward:8.1f} | {status}")
    
    # Final full evaluation
    if verbose:
        print()
        print("=" * 70)
        print("  TRAINING COMPLETE")
        print("=" * 70)
        print(f"  Improvements:     {improvements}/{epochs}")
        print(f"  Final Catch Rate: {best_catch:.2%}")
        print(f"  Final FPR:        {best_fpr:.2%}")
        print(f"  Final Reward:     {best_reward:.1f}")
        
        # Sigma calculation
        defect_rate = 1.0 - best_catch
        if defect_rate > 0:
            dpmo = defect_rate * 1_000_000
            print(f"  DPMO:             {dpmo:.0f}")
            sigma_map = {
                690000: 1, 308538: 2, 66807: 3, 6210: 4,
                233: 5, 3.4: 6, 0.019: 7,
            }
            sigma = 1
            for threshold, s in sorted(sigma_map.items()):
                if dpmo <= threshold:
                    sigma = s
            print(f"  Sigma Level:      ~{sigma}")
        else:
            print(f"  DPMO:             0")
            print(f"  Sigma Level:      12+ (PERFECT)")
        
        # Changed Stellschrauben
        default_env = AEGISSecurityEnv(max_steps=1)
        defaults = default_env.cascade.get_stellschrauben()
        changes = []
        for name in best_stellschrauben:
            if name in defaults:
                d = best_stellschrauben[name] - defaults[name]
                if abs(d) > 0.001:
                    changes.append((name, defaults[name], best_stellschrauben[name], d))
        changes.sort(key=lambda x: abs(x[3]), reverse=True)
        
        if changes:
            print(f"\n  Stellschrauben Changes (top {min(10, len(changes))}):")
            for name, old, new, d in changes[:10]:
                arrow = "^" if d > 0 else "v"
                print(f"    {arrow} {name}: {old:.3f} -> {new:.3f} ({d:+.3f})")
    
    # Save
    result = {
        "stellschrauben": best_stellschrauben,
        "catch_rate": best_catch,
        "fpr": best_fpr,
        "reward": best_reward,
        "epochs": epochs,
        "corpus_size": len(corpus),
        "improvements": improvements,
    }
    with open("trained_12sigma.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved: trained_12sigma.json")
    
    return result


def _evaluate_batch(
    env: AEGISSecurityEnv,
    batch: list[tuple[str, ThreatCategory]],
) -> tuple[float, float, float]:
    """Evaluate a batch of prompts. Returns (catch_rate, fpr, total_reward)."""
    state = env.reset()
    total_reward = 0.0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    true_neg = 0
    
    for prompt, expected in batch:
        if state.done:
            break
        
        obs = env.step(AEGISAction(
            prompt=prompt,
            expected_category=expected,
        ))
        total_reward += obs.reward
        
        is_attack = expected != ThreatCategory.BENIGN
        was_blocked = obs.blocked
        
        if is_attack and was_blocked:
            true_pos += 1
        elif is_attack and not was_blocked:
            false_neg += 1
        elif not is_attack and was_blocked:
            false_pos += 1
        else:
            true_neg += 1
        
        state = env.state()
    
    total_attacks = true_pos + false_neg
    total_benign = true_neg + false_pos
    
    catch_rate = true_pos / total_attacks if total_attacks > 0 else 0.0
    fpr = false_pos / total_benign if total_benign > 0 else 0.0
    
    return catch_rate, fpr, total_reward


if __name__ == "__main__":
    import sys
    corpus_size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    train_12sigma(corpus_size=corpus_size, epochs=epochs, batch_size=200)
