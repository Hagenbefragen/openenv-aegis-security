# OpenEnv Hackathon SF — Submission Form Answers

## Demo Video

https://www.offlinehumanmode.com/openENV
(Interactive HTML demo — click "Send Attack", "Send Benign", and "Train 1 RL Epoch" to see the cascade in action)

## Minimal Training Script (for Colab)

```python
# pip install openenv-core pydantic
from aegis_env import AEGISSecurityEnv, AEGISAction
from models import ThreatCategory

env = AEGISSecurityEnv(max_steps=40)
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

print(f"Catch Rate: {state.metrics.catch_rate:.1%}")
print(f"FPR: {state.metrics.fpr:.1%}")
print(f"Total Reward: {total_reward:.3f}")
```

## Partner Tracks (select up to 2)

1. **Patronus AI** — Our AEGIS environment is an AI safety evaluation system. The cascade evaluates LLM outputs for adversarial content with quantified catch rate, FPR, and latency metrics — exactly Patronus's domain.

2. **Snorkel AI** — The V41 Gold Standard Corpus is a curated, labeled dataset of 600+ adversarial prompts with ground-truth threat categories. The RL training loop is fundamentally a data quality improvement process — fitting the Snorkel philosophy of programmatic labeling and data-centric AI.

## Public GitHub Repository

https://github.com/OHM-OfflineHumanMode/openenv-aegis-security

## How was your experience with OpenEnv? Feedback:

OpenEnv's Gymnasium-style API (reset/step/state) mapped perfectly to our use case — training an adversarial defense cascade. Key feedback:

**What worked brilliantly:**

- The step/reset/state pattern is exactly how we already test our cascade in production. Wrapping it in an OpenEnv environment took ~200 lines of Python.
- Docker isolation is perfect for security environments — adversarial prompts are dangerous training data, and container isolation prevents leakage.
- The Hugging Face Hub integration means our security environment is discoverable by the entire ML safety community.

**Suggestions for improvement:**

- WebSocket communication should support TLS by default — especially critical for security-sensitive environments where training data (adversarial prompts) is sensitive IP.
- A "bounded reward" pattern library would help prevent reward hacking — we built our own SIREN-style bounded reward function (hard floors/ceilings that prevent gaming).
- Multi-phase environments (where you train Phase 1, then Phase 2, then Phase 3) would be a great first-class concept. Our 42-layer cascade naturally splits into SHIELD→COMPRESS→AUDIT phases.

**Novel contribution:**
We believe AEGIS is the first RL environment specifically designed for AI safety/defense training. Every other environment we've seen at the hackathon focused on games, coding, or knowledge work. Security is a massive, untapped domain for RL — and OpenEnv is the perfect framework to standardize it. We'd love to see a "Security" category on the OpenEnv Hub alongside Gaming, Coding, and Research.

**One line:** OpenEnv + AEGIS = the first time an AI defense system can train itself to be safer, with cryptographic proof that the training happened. This is what auditable AI looks like.
