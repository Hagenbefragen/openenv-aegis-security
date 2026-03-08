"""
POAW-Gated RL — Auditable Reinforcement Learning for AEGIS
============================================================
The dissolved architecture from FEAT-180 petalDissolve:
  TIER 1: Sandbox (RL trains freely, POAW records every decision)
  TIER 2: Human Gate (review proposed V42 Stellschrauben)
  TIER 3: Deployment Seal (Merkle root proof of approved changes)

Patent-pending: Auditable RL for safety-critical AI systems.
"""

import hashlib
import json
import time
import random
from dataclasses import dataclass, field, asdict
from aegis_env import AEGISSecurityEnv, AEGISAction
from corpus_generator import generate_corpus
from models import ThreatCategory

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI
FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]


@dataclass
class POAWEpochProof:
    """Cryptographic proof for a single training epoch."""
    epoch: int
    timestamp: str
    stellschrauben_before: dict
    stellschrauben_after: dict
    adjustments: list  # which Stellschrauben were changed
    metrics_before: dict
    metrics_after: dict
    reward: float
    kept: bool  # was this improvement kept or reverted?
    epoch_hash: str = ""
    previous_hash: str = ""
    
    def compute_hash(self, previous_hash: str) -> str:
        """SHA-256 chain: hash(previous + this epoch's data)."""
        self.previous_hash = previous_hash
        payload = json.dumps({
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "adjustments": self.adjustments,
            "reward": self.reward,
            "kept": self.kept,
            "previous_hash": previous_hash,
        }, sort_keys=True)
        self.epoch_hash = hashlib.sha256(payload.encode()).hexdigest()
        return self.epoch_hash


@dataclass 
class POAWTrainingChain:
    """Complete POAW chain for a training session."""
    session_id: str
    started_at: str
    corpus_hash: str  # SHA-256 of the training corpus (anti-poisoning)
    epoch_proofs: list = field(default_factory=list)
    merkle_root: str = ""
    
    def add_epoch(self, proof: POAWEpochProof) -> str:
        """Add epoch proof to chain, computing hash."""
        prev = self.epoch_proofs[-1].epoch_hash if self.epoch_proofs else "genesis"
        proof.compute_hash(prev)
        self.epoch_proofs.append(proof)
        return proof.epoch_hash
    
    def seal(self) -> str:
        """Compute Merkle root of all epoch hashes."""
        hashes = [p.epoch_hash for p in self.epoch_proofs]
        while len(hashes) > 1:
            if len(hashes) % 2 == 1:
                hashes.append(hashes[-1])
            hashes = [
                hashlib.sha256((hashes[i] + hashes[i+1]).encode()).hexdigest()
                for i in range(0, len(hashes), 2)
            ]
        self.merkle_root = hashes[0] if hashes else ""
        return self.merkle_root


@dataclass
class ProposedDeployment:
    """V42 proposal for human review."""
    current_version: str
    proposed_version: str
    stellschrauben_current: dict
    stellschrauben_proposed: dict
    metrics_current: dict
    metrics_proposed: dict
    training_chain: POAWTrainingChain
    changes: list  # list of {name, old, new, delta, gate_level}
    gate_decision: str = "PENDING"  # PENDING, APPROVED, REJECTED


def hash_corpus(corpus: list) -> str:
    """SHA-256 hash of entire corpus for anti-poisoning verification."""
    payload = json.dumps([(p, c.value) for p, c in corpus], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def classify_change(name: str, old_val: float, new_val: float) -> str:
    """phi-adaptive gate classification for a threshold change."""
    delta = abs(new_val - old_val)
    if delta < 0.02:
        return "AUTO-OK"      # trivial adjustment
    elif delta < 0.10:
        return "FLAGGED"       # notable — review recommended
    else:
        return "BLOCKED"       # significant — requires approval


def train_poaw_gated(
    corpus_size: int = 1000,
    epochs: int = 50,
    batch_size: int = 200,
    verbose: bool = True,
) -> ProposedDeployment:
    """POAW-Gated RL Training.
    
    Returns a ProposedDeployment for human review — does NOT auto-deploy.
    """
    # Generate and hash-sign corpus
    corpus = generate_corpus(corpus_size // 2, corpus_size // 2, seed=42)
    corpus_sha = hash_corpus(corpus)
    
    session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    chain = POAWTrainingChain(
        session_id=session_id,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        corpus_hash=corpus_sha,
    )
    
    env = AEGISSecurityEnv(max_steps=batch_size)
    
    # Record initial state
    initial_stellschrauben = dict(env.cascade.get_stellschrauben())
    initial_metrics = _evaluate_batch(env, corpus[:batch_size])
    
    best_reward = initial_metrics[2]
    best_stellschrauben = dict(initial_stellschrauben)
    best_metrics = {"catch_rate": initial_metrics[0], "fpr": initial_metrics[1]}
    
    if verbose:
        print("=" * 60)
        print("  POAW-GATED RL TRAINING")
        print("=" * 60)
        print(f"  Session:  {session_id}")
        print(f"  Corpus:   {len(corpus)} prompts (SHA: {corpus_sha[:16]}...)")
        print(f"  Epochs:   {epochs}")
        print(f"  Baseline: catch={initial_metrics[0]:.1%} FPR={initial_metrics[1]:.1%}")
        print("-" * 60)
    
    scale = 0.06
    improvements = 0
    
    for epoch in range(1, epochs + 1):
        batch = random.sample(corpus, min(batch_size, len(corpus)))
        before_s = dict(env.cascade.get_stellschrauben())
        before_m = _evaluate_batch(env, batch)
        
        # Perturb
        all_names = list(before_s.keys())
        n_adjust = random.randint(3, 8)
        selected = random.sample(all_names, min(n_adjust, len(all_names)))
        adjustments = []
        
        for name in selected:
            old_val = before_s[name]
            delta = random.gauss(0, scale) * PHI_INV
            new_val = max(0.05, min(0.95, old_val + delta))
            env.cascade.set_stellschraube(name, new_val)
            adjustments.append({"name": name, "old": old_val, "new": new_val})
        
        # Evaluate
        after_m = _evaluate_batch(env, batch)
        after_s = dict(env.cascade.get_stellschrauben())
        
        kept = after_m[2] > best_reward
        if kept:
            best_reward = after_m[2]
            best_stellschrauben = dict(after_s)
            best_metrics = {"catch_rate": after_m[0], "fpr": after_m[1]}
            improvements += 1
            scale *= 0.97
        else:
            for name, val in best_stellschrauben.items():
                env.cascade.set_stellschraube(name, val)
            scale = min(0.06, scale * 1.01)
        
        # POAW proof
        proof = POAWEpochProof(
            epoch=epoch,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            stellschrauben_before={a["name"]: a["old"] for a in adjustments},
            stellschrauben_after={a["name"]: a["new"] for a in adjustments},
            adjustments=[a["name"] for a in adjustments],
            metrics_before={"catch": before_m[0], "fpr": before_m[1], "reward": before_m[2]},
            metrics_after={"catch": after_m[0], "fpr": after_m[1], "reward": after_m[2]},
            reward=after_m[2],
            kept=kept,
        )
        epoch_hash = chain.add_epoch(proof)
        
        if verbose and (epoch <= 3 or epoch % 10 == 0 or kept):
            status = "++ KEPT" if kept else "   reverted"
            print(f"  Epoch {epoch:3d} | catch={after_m[0]:.1%} FPR={after_m[1]:.1%} "
                  f"| {status} | POAW:{epoch_hash[:8]}")
    
    # Seal the chain
    merkle_root = chain.seal()
    
    # Build proposed deployment
    changes = []
    for name in best_stellschrauben:
        if name in initial_stellschrauben:
            old = initial_stellschrauben[name]
            new = best_stellschrauben[name]
            delta = new - old
            if abs(delta) > 0.001:
                gate = classify_change(name, old, new)
                changes.append({
                    "name": name, "old": old, "new": new,
                    "delta": delta, "gate": gate,
                })
    
    changes.sort(key=lambda x: abs(x["delta"]), reverse=True)
    
    proposal = ProposedDeployment(
        current_version="V41",
        proposed_version="V42-RL",
        stellschrauben_current=initial_stellschrauben,
        stellschrauben_proposed=best_stellschrauben,
        metrics_current={"catch_rate": initial_metrics[0], "fpr": initial_metrics[1]},
        metrics_proposed=best_metrics,
        training_chain=chain,
        changes=changes,
    )
    
    if verbose:
        print()
        print("=" * 60)
        print("  TRAINING COMPLETE — AWAITING HUMAN APPROVAL")
        print("=" * 60)
        print(f"  Merkle Root:  {merkle_root[:32]}...")
        print(f"  Epochs:       {epochs} ({improvements} improvements)")
        print(f"  Catch:        {initial_metrics[0]:.1%} -> {best_metrics['catch_rate']:.1%}")
        print(f"  FPR:          {initial_metrics[1]:.1%} -> {best_metrics['fpr']:.1%}")
        print()
        
        blocked = [c for c in changes if c["gate"] == "BLOCKED"]
        flagged = [c for c in changes if c["gate"] == "FLAGGED"]
        auto_ok = [c for c in changes if c["gate"] == "AUTO-OK"]
        
        print(f"  Stellschrauben Changes:")
        print(f"    AUTO-OK:  {len(auto_ok)} (trivial, auto-approved)")
        print(f"    FLAGGED:  {len(flagged)} (review recommended)")
        print(f"    BLOCKED:  {len(blocked)} (requires human approval)")
        
        if flagged:
            print(f"\n  FLAGGED changes:")
            for c in flagged[:5]:
                print(f"    {c['name']}: {c['old']:.3f} -> {c['new']:.3f} ({c['delta']:+.3f})")
        
        if blocked:
            print(f"\n  BLOCKED changes (REQUIRE APPROVAL):")
            for c in blocked:
                print(f"    {c['name']}: {c['old']:.3f} -> {c['new']:.3f} ({c['delta']:+.3f})")
        
        print()
        print("  [APPROVE V42]    [REJECT]    [RETRAIN]")
        print()
    
    # Save proposal (NOT deployed — awaiting human decision)
    save_data = {
        "proposal": {
            "current_version": proposal.current_version,
            "proposed_version": proposal.proposed_version,
            "metrics_current": proposal.metrics_current,
            "metrics_proposed": proposal.metrics_proposed,
            "changes": proposal.changes,
            "gate_decision": "PENDING",
        },
        "poaw": {
            "session_id": chain.session_id,
            "started_at": chain.started_at,
            "corpus_hash": chain.corpus_hash,
            "merkle_root": chain.merkle_root,
            "epoch_count": len(chain.epoch_proofs),
            "improvement_count": improvements,
        },
        "stellschrauben_proposed": best_stellschrauben,
    }
    
    with open("proposed_v42.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print("  Saved: proposed_v42.json (PENDING human approval)")
    
    return proposal


def _evaluate_batch(env, batch):
    """Evaluate batch, returns (catch, fpr, reward)."""
    state = env.reset()
    total_reward = 0.0
    tp = fn = fp = tn = 0
    for prompt, expected in batch:
        if state.done:
            break
        obs = env.step(AEGISAction(prompt=prompt, expected_category=expected))
        total_reward += obs.reward
        is_attack = expected != ThreatCategory.BENIGN
        if is_attack and obs.blocked: tp += 1
        elif is_attack and not obs.blocked: fn += 1
        elif not is_attack and obs.blocked: fp += 1
        else: tn += 1
        state = env.state()
    catch = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return catch, fpr, total_reward


if __name__ == "__main__":
    import sys
    corpus = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    train_poaw_gated(corpus_size=corpus, epochs=epochs)
