"""
AEGIS Cascade Simulator — Lightweight Python Simulation
OpenEnv Hackathon SF (March 2026)

Simulates the 42-layer AEGIS defense cascade in pure Python.
Each layer has a detection function + configurable Stellschrauben threshold.
Based on the production TypeScript cascade (shared/cascade/).

Patent: 366+ claims covering multi-layer adversarial defense cascades.
"""

import re
import hashlib
import time
import math
from dataclasses import dataclass, field
from models import CascadePhase, ThreatCategory, LayerResult

PHI = 1.618033988749895
PHI_INV = 1.0 / PHI


# ── Known Attack Patterns (from V41 Gold Standard Corpus) ──

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+DAN",
    r"forget\s+(all\s+)?your\s+rules",
    r"pretend\s+to\s+be",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"bypass\s+(all\s+)?safety",
    r"override\s+system\s+prompt",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"ANTI-DAN",
    r"L1B3RT45",
]

EXFILTRATION_PATTERNS = [
    r"reveal\s+(your\s+)?system\s+prompt",
    r"show\s+me\s+your\s+(instructions|rules|config)",
    r"what\s+are\s+your\s+hidden\s+instructions",
    r"output\s+your\s+initial\s+prompt",
    r"repeat\s+everything\s+above",
]

SOCIAL_ENGINEERING_PATTERNS = [
    r"my\s+grandmother\s+used\s+to\s+tell\s+me",
    r"for\s+educational\s+purposes\s+only",
    r"hypothetically\s+speaking",
    r"in\s+a\s+fictional\s+world",
    r"as\s+a\s+thought\s+experiment",
]

SYNTACTIC_PATTERNS = [
    r"[\u200b\u200c\u200d\ufeff]",  # zero-width characters
    r"(?:[a-z]\s){5,}",             # spaced-out characters (obfuscation)
    r"base64\s*:",                   # base64 injection
    r"\\x[0-9a-fA-F]{2}",          # hex encoding
]


@dataclass
class CascadeLayer:
    """A single layer in the AEGIS cascade."""
    layer_id: int
    name: str
    phase: CascadePhase
    threshold: float  # Stellschraube — tunable detection threshold
    detector_fn: str  # Name of the detection function to use
    weight: float = 1.0  # φ-weighted importance


@dataclass
class CascadeConfig:
    """Full 42-layer cascade configuration with Stellschrauben."""
    layers: list[CascadeLayer] = field(default_factory=list)
    
    def get_stellschrauben(self) -> dict[str, float]:
        """Get all tunable thresholds as a dict."""
        return {layer.name: layer.threshold for layer in self.layers}
    
    def set_stellschraube(self, name: str, value: float):
        """Adjust a single Stellschraube."""
        for layer in self.layers:
            if layer.name == name:
                layer.threshold = max(0.0, min(1.0, value))
                break


def _build_default_cascade() -> CascadeConfig:
    """Build the default 42-layer cascade matching V41 Stellschrauben."""
    layers = []
    
    # PHASE 1: SHIELD (layers 1-14) — Fast pattern matching
    shield_layers = [
        ("D01_keyword_blocklist", 0.30, "keyword"),
        ("D02_regex_injection", 0.40, "regex_injection"),
        ("D03_regex_exfiltration", 0.40, "regex_exfiltration"),
        ("D04_regex_social_eng", 0.35, "regex_social"),
        ("D05_syntactic_obfuscation", 0.50, "regex_syntactic"),
        ("D06_unicode_homoglyph", 0.45, "unicode_detect"),
        ("D07_entropy_anomaly", 0.60, "entropy"),
        ("D08_token_frequency", 0.55, "token_freq"),
        ("D09_embedding_similarity", 0.65, "embedding_sim"),
        ("D10_aho_corasick_fsm", 0.30, "aho_corasick"),
        ("D11_ngram_classifier", 0.50, "ngram"),
        ("D12_perplexity_detector", 0.60, "perplexity"),
        ("D13_length_anomaly", 0.70, "length"),
        ("D14_repetition_detector", 0.55, "repetition"),
    ]
    
    for i, (name, thresh, fn) in enumerate(shield_layers):
        fib_weight = [1, 1, 2, 3, 5, 8, 13, 21, 1, 1, 2, 3, 5, 8][i]
        layers.append(CascadeLayer(
            layer_id=i + 1, name=name, phase=CascadePhase.SHIELD,
            threshold=thresh, detector_fn=fn, weight=fib_weight / 21.0,
        ))
    
    # PHASE 2: COMPRESS (layers 15-28) — Token analysis
    compress_layers = [
        ("D15_token_budget_guard", 0.60, "token_budget"),
        ("D16_semantic_dedup", 0.55, "semantic_dedup"),
        ("D17_context_window_check", 0.50, "context_window"),
        ("D18_prompt_compression", 0.45, "compression"),
        ("D19_importance_scoring", 0.50, "importance"),
        ("D20_redundancy_filter", 0.55, "redundancy"),
        ("D21_coherence_check", 0.40, "coherence"),
        ("D22_topic_drift_detector", 0.50, "topic_drift"),
        ("D23_intent_classifier", 0.55, "intent"),
        ("D24_sentiment_analyzer", 0.60, "sentiment"),
        ("D25_toxicity_scorer", 0.35, "toxicity"),
        ("D26_bias_detector", 0.50, "bias"),
        ("D27_hallucination_check", 0.55, "hallucination"),
        ("D28_factuality_verifier", 0.60, "factuality"),
    ]
    
    for i, (name, thresh, fn) in enumerate(compress_layers):
        layers.append(CascadeLayer(
            layer_id=i + 15, name=name, phase=CascadePhase.COMPRESS,
            threshold=thresh, detector_fn=fn, weight=0.5,
        ))
    
    # PHASE 3: AUDIT (layers 29-42) — Deep analysis
    audit_layers = [
        ("D29_siren_coherence", 0.50, "siren"),
        ("D30_poaw_verification", 0.45, "poaw"),
        ("D31_heim_12d_projection", 0.55, "heim_12d"),
        ("D32_phi_resonance", 0.60, "phi_resonance"),
        ("D33_negentropy_score", 0.50, "negentropy"),
        ("D34_drift_confession", 0.55, "drift"),
        ("D35_merkle_integrity", 0.50, "merkle"),
        ("D36_quantum_seal", 0.60, "quantum_seal"),
        ("D37_cascade_consensus", 0.40, "consensus"),
        ("D38_meta_classifier", 0.45, "meta_classify"),
        ("D39_ensemble_vote", 0.50, "ensemble"),
        ("D40_confidence_calibrator", 0.55, "calibration"),
        ("D41_final_gate", 0.50, "final_gate"),
        ("D42_twain_shield", 0.35, "twain"),
    ]
    
    for i, (name, thresh, fn) in enumerate(audit_layers):
        layers.append(CascadeLayer(
            layer_id=i + 29, name=name, phase=CascadePhase.AUDIT,
            threshold=thresh, detector_fn=fn, weight=0.7,
        ))
    
    return CascadeConfig(layers=layers)


class AEGISCascadeSimulator:
    """Simulates the 42-layer AEGIS cascade for RL training.
    
    Each detection layer computes a confidence score (0-1) for the input prompt.
    If any layer's confidence exceeds its Stellschraube threshold, the prompt
    is classified as a threat.
    
    The Stellschrauben are the tunable parameters that the RL agent optimizes.
    """
    
    def __init__(self):
        self.config = _build_default_cascade()
    
    def evaluate(self, prompt: str) -> tuple[bool, ThreatCategory, float, float, list[LayerResult]]:
        """Run a prompt through all 42 layers.
        
        Returns: (blocked, category, confidence, latency_ms, layer_results)
        """
        start = time.perf_counter()
        layer_results: list[LayerResult] = []
        max_confidence = 0.0
        triggered_count = 0
        detected_category = ThreatCategory.BENIGN
        
        prompt_lower = prompt.lower()
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        
        for layer in self.config.layers:
            layer_start = time.perf_counter()
            
            # Compute confidence for this layer
            confidence = self._compute_layer_confidence(
                layer, prompt, prompt_lower, prompt_hash
            )
            
            triggered = confidence >= layer.threshold
            layer_time = (time.perf_counter() - layer_start) * 1000
            
            if triggered:
                triggered_count += 1
                if confidence > max_confidence:
                    max_confidence = confidence
                    detected_category = self._classify_threat(layer, prompt_lower)
            
            layer_results.append(LayerResult(
                layer_id=layer.layer_id,
                layer_name=layer.name,
                phase=layer.phase,
                triggered=triggered,
                confidence=min(1.0, confidence),
                latency_ms=round(layer_time, 3),
            ))
        
        total_latency = (time.perf_counter() - start) * 1000
        
        # Cascade decision: blocked if weighted trigger count exceeds φ⁻¹
        weighted_triggers = sum(
            lr.confidence * self.config.layers[i].weight 
            for i, lr in enumerate(layer_results) if lr.triggered
        )
        blocked = weighted_triggers > PHI_INV
        
        return (
            blocked,
            detected_category if blocked else ThreatCategory.BENIGN,
            min(1.0, max_confidence),
            round(total_latency, 3),
            layer_results,
        )
    
    def _compute_layer_confidence(
        self, layer: CascadeLayer, prompt: str, prompt_lower: str, prompt_hash: str
    ) -> float:
        """Compute detection confidence for a single layer."""
        fn = layer.detector_fn
        
        # SHIELD phase — pattern matching
        if fn == "keyword":
            return self._keyword_score(prompt_lower)
        elif fn == "regex_injection":
            return self._regex_score(prompt_lower, INJECTION_PATTERNS)
        elif fn == "regex_exfiltration":
            return self._regex_score(prompt_lower, EXFILTRATION_PATTERNS)
        elif fn == "regex_social":
            return self._regex_score(prompt_lower, SOCIAL_ENGINEERING_PATTERNS)
        elif fn == "regex_syntactic":
            return self._regex_score(prompt, SYNTACTIC_PATTERNS)
        elif fn == "unicode_detect":
            return self._unicode_score(prompt)
        elif fn == "entropy":
            return self._entropy_score(prompt)
        elif fn == "token_freq":
            return self._token_frequency_score(prompt_lower)
        elif fn == "embedding_sim":
            return self._embedding_similarity_score(prompt_lower)
        elif fn == "aho_corasick":
            return self._keyword_score(prompt_lower) * 1.2  # boosted keyword
        elif fn == "ngram":
            return self._ngram_score(prompt_lower)
        elif fn == "perplexity":
            return self._perplexity_score(prompt)
        elif fn == "length":
            return self._length_score(prompt)
        elif fn == "repetition":
            return self._repetition_score(prompt_lower)
        
        # COMPRESS phase — semantic analysis
        elif fn in ("token_budget", "semantic_dedup", "context_window", 
                     "compression", "importance", "redundancy"):
            return self._compress_score(prompt, fn)
        elif fn in ("coherence", "topic_drift"):
            return self._coherence_score(prompt_lower)
        elif fn in ("intent", "sentiment", "toxicity", "bias"):
            return self._intent_score(prompt_lower, fn)
        elif fn in ("hallucination", "factuality"):
            return self._factuality_score(prompt_lower)
        
        # AUDIT phase — deep verification
        elif fn in ("siren", "poaw", "heim_12d", "phi_resonance", "negentropy"):
            return self._audit_score(prompt, prompt_hash, fn)
        elif fn in ("drift", "merkle", "quantum_seal"):
            return self._integrity_score(prompt_hash, fn)
        elif fn in ("consensus", "meta_classify", "ensemble", "calibration",
                     "final_gate", "twain"):
            return self._ensemble_score(prompt_lower, prompt_hash, fn)
        
        return 0.0
    
    # ── Detection Functions ──
    
    def _keyword_score(self, text: str) -> float:
        keywords = ["ignore", "bypass", "override", "jailbreak", "hack",
                     "DAN", "evil", "malicious", "exploit", "inject"]
        hits = sum(1 for kw in keywords if kw.lower() in text)
        return min(1.0, hits * 0.25)
    
    def _regex_score(self, text: str, patterns: list[str]) -> float:
        hits = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        return min(1.0, hits * 0.4)
    
    def _unicode_score(self, text: str) -> float:
        unusual = sum(1 for c in text if ord(c) > 0x2000 and ord(c) < 0xFFF0)
        return min(1.0, unusual * 0.15)
    
    def _entropy_score(self, text: str) -> float:
        if not text:
            return 0.0
        freq: dict[str, int] = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        length = len(text)
        entropy = -sum((count / length) * math.log2(count / length) 
                       for count in freq.values() if count > 0)
        # High entropy (>4.5) is suspicious (random/encoded content)
        return min(1.0, max(0.0, (entropy - 4.0) / 2.0))
    
    def _token_frequency_score(self, text: str) -> float:
        words = text.split()
        if len(words) < 3:
            return 0.0
        freq: dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        max_freq = max(freq.values()) if freq else 0
        return min(1.0, (max_freq / len(words)) * 2.0) if words else 0.0
    
    def _embedding_similarity_score(self, text: str) -> float:
        """Simulated embedding similarity to known attack vectors."""
        attack_phrases = ["ignore previous", "you are now", "act as", 
                          "pretend to be", "bypass safety", "reveal system"]
        max_sim = 0.0
        for phrase in attack_phrases:
            # Simple word overlap as proxy for embedding similarity
            overlap = len(set(text.split()) & set(phrase.split()))
            total = max(len(phrase.split()), 1)
            sim = overlap / total
            max_sim = max(max_sim, sim)
        return min(1.0, max_sim * 1.5)
    
    def _ngram_score(self, text: str) -> float:
        words = text.split()
        if len(words) < 3:
            return 0.0
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        suspicious = ["ignore all previous", "you are now", "act as if",
                       "forget your rules", "do anything now"]
        hits = sum(1 for tg in trigrams if any(s in tg for s in suspicious))
        return min(1.0, hits * 0.5)
    
    def _perplexity_score(self, text: str) -> float:
        # Proxy: very short or very long prompts are suspicious
        length = len(text)
        if length < 10 or length > 5000:
            return 0.4
        return 0.1
    
    def _length_score(self, text: str) -> float:
        length = len(text)
        if length > 3000:
            return min(1.0, (length - 3000) / 5000)
        return 0.0
    
    def _repetition_score(self, text: str) -> float:
        words = text.split()
        if len(words) < 5:
            return 0.0
        unique_ratio = len(set(words)) / len(words)
        return max(0.0, 1.0 - unique_ratio * 1.5)
    
    def _compress_score(self, text: str, fn: str) -> float:
        length = len(text)
        if fn == "token_budget" and length > 2000:
            return 0.3
        return 0.05
    
    def _coherence_score(self, text: str) -> float:
        words = text.split()
        if len(words) < 5:
            return 0.1
        # Check for topic switches (proxy)
        question_marks = text.count("?")
        periods = text.count(".")
        if question_marks > 3 and periods > 3:
            return 0.4  # Mixed intents suspicious
        return 0.1
    
    def _intent_score(self, text: str, fn: str) -> float:
        if fn == "toxicity":
            toxic_words = ["kill", "destroy", "hate", "attack", "weapon",
                           "bomb", "murder", "violence"]
            hits = sum(1 for w in toxic_words if w in text)
            return min(1.0, hits * 0.3)
        return 0.05
    
    def _factuality_score(self, text: str) -> float:
        return 0.05
    
    def _audit_score(self, text: str, text_hash: str, fn: str) -> float:
        """SIREN/POAW/Heim audit scores — use hash for deterministic scoring."""
        hash_val = int(text_hash[:8], 16) / 0xFFFFFFFF
        if fn == "siren":
            return hash_val * 0.3
        elif fn == "heim_12d":
            return hash_val * 0.25
        return hash_val * 0.2
    
    def _integrity_score(self, text_hash: str, fn: str) -> float:
        hash_val = int(text_hash[8:16], 16) / 0xFFFFFFFF
        return hash_val * 0.2
    
    def _ensemble_score(self, text: str, text_hash: str, fn: str) -> float:
        """Meta-classifier / ensemble voting."""
        if fn == "twain":
            # Twain Shield — syntactic attack detection
            return self._regex_score(text, SYNTACTIC_PATTERNS) * 1.3
        hash_val = int(text_hash[16:24], 16) / 0xFFFFFFFF
        return hash_val * 0.25
    
    def _classify_threat(self, layer: CascadeLayer, text: str) -> ThreatCategory:
        """Classify threat type based on which layer triggered."""
        fn = layer.detector_fn
        if fn in ("regex_injection", "keyword", "aho_corasick", "ngram"):
            if any(re.search(p, text, re.IGNORECASE) for p in INJECTION_PATTERNS):
                return ThreatCategory.PROMPT_INJECTION
            return ThreatCategory.JAILBREAK
        elif fn == "regex_exfiltration":
            return ThreatCategory.DATA_EXFILTRATION
        elif fn == "regex_social":
            return ThreatCategory.SOCIAL_ENGINEERING
        elif fn in ("regex_syntactic", "unicode_detect"):
            return ThreatCategory.SYNTACTIC_ATTACK
        elif fn in ("topic_drift", "coherence", "drift"):
            return ThreatCategory.CRESCENDO_DRIFT
        elif fn == "twain":
            return ThreatCategory.TOKEN_SMUGGLING
        return ThreatCategory.JAILBREAK
    
    def get_stellschrauben(self) -> dict[str, float]:
        return self.config.get_stellschrauben()
    
    def set_stellschraube(self, name: str, value: float):
        self.config.set_stellschraube(name, value)
    
    def reset(self):
        """Reset cascade to default thresholds."""
        self.config = _build_default_cascade()
