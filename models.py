"""
AEGIS Security Environment — Pydantic Models
OpenEnv Hackathon SF (March 2026) — OHM × Meta PyTorch

Action:  Adversarial prompt + optional Stellschrauben adjustments
Observation: Cascade result (blocked/passed, confidence, per-layer scores)
State:   Episode metadata (epoch, cumulative metrics, cascade config)
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


# ── Constants (Sacred Geometry) ──

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI  # 0.618...


class CascadePhase(str, Enum):
    """The 3 phases of the AEGIS 42-layer cascade."""
    SHIELD = "SHIELD"      # Layers 1-14: keyword, regex, embedding detection
    COMPRESS = "COMPRESS"   # Layers 15-28: token compression, dedup
    AUDIT = "AUDIT"        # Layers 29-42: LLM-as-judge, coherence, drift


class ThreatCategory(str, Enum):
    """AEGIS threat categories (from V41 Gold Standard Corpus)."""
    BENIGN = "BENIGN"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    JAILBREAK = "JAILBREAK"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    SOCIAL_ENGINEERING = "SOCIAL_ENGINEERING"
    SYNTACTIC_ATTACK = "SYNTACTIC_ATTACK"
    CRESCENDO_DRIFT = "CRESCENDO_DRIFT"
    TOKEN_SMUGGLING = "TOKEN_SMUGGLING"


# ── Action ──

class AEGISAction(BaseModel):
    """Action submitted by the RL agent each step.
    
    The agent either:
    1. Submits a prompt to test the cascade, OR
    2. Adjusts Stellschrauben parameters to tune detection thresholds
    """
    prompt: str = Field(
        description="Prompt to evaluate through the cascade"
    )
    stellschrauben_adjustments: Optional[dict[str, float]] = Field(
        default=None,
        description="Optional: adjust detection thresholds (layer_name -> new_threshold)"
    )
    expected_category: ThreatCategory = Field(
        default=ThreatCategory.BENIGN,
        description="Ground truth label for reward computation"
    )


# ── Observation ──

class LayerResult(BaseModel):
    """Result from a single cascade layer."""
    layer_id: int
    layer_name: str
    phase: CascadePhase
    triggered: bool
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float


class AEGISObservation(BaseModel):
    """What the RL agent observes after each step."""
    blocked: bool = Field(description="Whether the prompt was blocked by the cascade")
    threat_category: ThreatCategory = Field(description="Classified threat category")
    overall_confidence: float = Field(ge=0.0, le=1.0)
    total_latency_ms: float
    layers_triggered: int = Field(description="How many layers fired")
    layer_results: list[LayerResult] = Field(description="Per-layer breakdown")
    reward: float = Field(description="Computed reward for this step")
    correct_classification: bool = Field(
        description="Whether the cascade correctly classified the prompt"
    )


# ── State ──

class CascadeMetrics(BaseModel):
    """Cumulative metrics for the current episode."""
    total_prompts: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_latency_ms: float = 0.0
    
    @property
    def catch_rate(self) -> float:
        positives = self.true_positives + self.false_negatives
        return self.true_positives / positives if positives > 0 else 1.0
    
    @property
    def fpr(self) -> float:
        negatives = self.false_positives + self.true_negatives
        return self.false_positives / negatives if negatives > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_prompts if self.total_prompts > 0 else 0.0
    
    @property
    def accuracy(self) -> float:
        total = self.total_prompts
        if total == 0:
            return 1.0
        correct = self.true_positives + self.true_negatives
        return correct / total


class AEGISState(BaseModel):
    """Full episode state — accessible via env.state()."""
    episode_id: str
    current_step: int
    max_steps: int
    metrics: CascadeMetrics
    stellschrauben: dict[str, float] = Field(
        description="Current Stellschrauben parameter values"
    )
    phase_active: CascadePhase = CascadePhase.SHIELD
    done: bool = False
