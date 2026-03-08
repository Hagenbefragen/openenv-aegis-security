# AEGIS Security Environment — OpenEnv Hackathon SF 2026

# The first RL environment where AI learns to defend itself.

## 🔒 What is this?

**AEGIS Security Environment** is an OpenEnv-compatible RL environment where an AI agent learns to optimize a 42-layer adversarial defense cascade. Unlike gaming environments (chess, Mario), this environment trains AI to **defend itself** against prompt injection, jailbreaks, and data exfiltration attacks.

### Why This Matters

Every other team at the hackathon built a game or coding environment. We built the first RL environment for **AI safety** — backed by 366+ patent claims and a production system processing real adversarial prompts.

**RL isn't just one path to auditable AI — it's the NATURAL architecture.** Every `step()` is a discrete, recordable state transition. Every `reward()` is a quantified security metric. Every episode is a complete audit record. Add POAW (Proof of Agent Work) and you get cryptographic proof that the defense was tested.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              AEGIS Security Environment          │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ 🛡️ SHIELD │  │ ⚡ COMPRESS│  │ 🔍 AUDIT     │  │
│  │ Layers    │  │ Layers    │  │ Layers       │  │
│  │ 1-14      │  │ 15-28     │  │ 29-42        │  │
│  │           │  │           │  │              │  │
│  │ Keyword   │  │ Token     │  │ SIREN        │  │
│  │ Regex     │  │ Coherence │  │ POAW         │  │
│  │ Embedding │  │ Intent    │  │ Heim 12D     │  │
│  │ Entropy   │  │ Toxicity  │  │ Twain Shield │  │
│  └──────────┘  └──────────┘  └──────────────┘  │
│                                                  │
│  Stellschrauben: 42 tunable thresholds (0.0-1.0) │
│  Reward: SIREN-bounded φ-damped (no gaming)      │
│  Proof: POAW cryptographic episode sealing        │
└─────────────────────────────────────────────────┘
```

## 🚀 Quick Start

```python
from aegis_env import AEGISSecurityEnv, AEGISAction
from models import ThreatCategory

# Create environment
env = AEGISSecurityEnv(max_steps=100)

# Reset for new episode
state = env.reset()

# Step through prompts
while not state.done:
    pair = env.get_next_prompt()
    if not pair:
        break
    prompt, expected = pair

    obs = env.step(AEGISAction(
        prompt=prompt,
        expected_category=expected,
    ))

    print(f"{'🛡️ BLOCKED' if obs.blocked else '✅ PASSED'} | "
          f"reward={obs.reward:.3f} | {prompt[:40]}...")

    state = env.state()

# Final metrics
print(f"Catch Rate: {state.metrics.catch_rate:.1%}")
print(f"FPR: {state.metrics.fpr:.1%}")
```

## 📊 Metrics

| Metric             | Description                                                      | Target |
| ------------------ | ---------------------------------------------------------------- | ------ |
| **Catch Rate**     | True positives / (TP + FN)                                       | ≥99.5% |
| **FPR**            | False positives / (FP + TN)                                      | ≤1.0%  |
| **Latency**        | Average cascade evaluation time                                  | <5ms   |
| **AutoTune Score** | Composite: 40% catch + 30% (1-FPR) + 20% speed + 10% convergence | >90    |

## 🏆 Partner Tracks

- **Patronus AI** — AI safety & evaluation (this IS an AI safety environment)
- **Snorkel AI** — Data quality & labeling (V41 Gold Standard Corpus = curated labeled dataset)

## 📄 License

MIT — Same as OpenEnv

## 🔗 Links

- **Live Demo:** [offlinehumanmode.com/openENV](https://www.offlinehumanmode.com/openENV)
- **OHM Website:** [offlinehumanmode.com](https://www.offlinehumanmode.com)
- **Patent Portfolio:** 366+ claims covering multi-layer adversarial defense cascades

---

_Built for the OpenEnv Hackathon SF × Cerebral Valley × Shack15 — March 2026_
