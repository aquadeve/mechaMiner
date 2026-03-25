#!/usr/bin/env python3
"""
FSOT Core System - Consciousness State and Thought Engine
=========================================================

Implements the core FSOT (Few-Shot Operational Thought) consciousness model:

  C(t) = [attention_vector, confidence_levels, active_concepts,
           uncertainty_map, goal_state]

  C(t+1) = f(C(t), Input(t), Memory(t))

Thought scoring:
  score(thought_i) = C_t · W_i

Thought selection:
  P(thought_i) = softmax(score_i / temperature)
"""

import math
import random
import hashlib
import struct
import binascii
from typing import List, Dict, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility helpers (pure Python, no numpy required)
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    """Dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


def _softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    """Compute softmax with temperature scaling."""
    if temperature <= 0:
        temperature = 1e-8
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exps = [math.exp(s - max_s) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _norm(v: List[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def _entropy(v: List[float]) -> float:
    """Shannon entropy of a probability-like vector (clipped to [1e-12, 1])."""
    total = sum(abs(x) for x in v) or 1.0
    probs = [abs(x) / total for x in v]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


# ---------------------------------------------------------------------------
# Thought template
# ---------------------------------------------------------------------------

class Thought:
    """
    A reusable reasoning template embedded as a vector.

    Attributes
    ----------
    pattern   : Natural-language description of the reasoning rule
    confidence: Float in [0, 1] — how reliable this thought has been
    modality  : Domain/modality label (e.g. "numeric", "hash", "nonce")
    origin    : Where this thought came from
    embedding : High-dimensional vector used for scoring against C(t)
    """

    DIM = 64  # embedding dimensionality

    def __init__(
        self,
        pattern: str,
        confidence: float = 0.5,
        modality: str = "numeric",
        origin: str = "init",
        embedding: Optional[List[float]] = None,
    ) -> None:
        self.pattern = pattern
        self.confidence = max(0.0, min(1.0, confidence))
        self.modality = modality
        self.origin = origin
        # Deterministic embedding seeded from pattern hash so equal patterns
        # always produce the same embedding (reproducible across runs).
        if embedding is not None:
            self.embedding: List[float] = list(embedding)
        else:
            seed_int = int(hashlib.md5(pattern.encode()).hexdigest(), 16) % (2 ** 32)
            rng = random.Random(seed_int)
            self.embedding = [rng.gauss(0, 1) for _ in range(self.DIM)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self.pattern,
            "confidence": self.confidence,
            "modality": self.modality,
            "origin": self.origin,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Thought(pattern={self.pattern!r}, confidence={self.confidence:.2f},"
            f" modality={self.modality!r})"
        )


# ---------------------------------------------------------------------------
# Consciousness State
# ---------------------------------------------------------------------------

class ConsciousnessState:
    """
    C(t) ∈ R^n  — the system's current awareness snapshot.

    Sub-vectors
    -----------
    attention_vector   : Where compute is focused
    confidence_levels  : Per-sub-vector confidence
    active_concepts    : Activated knowledge nodes
    uncertainty_map    : Epistemic uncertainty per concept
    goal_state         : Current goal encoding
    """

    SEGMENT = 16  # per-sub-vector dimensionality
    TOTAL_DIM = SEGMENT * 5  # 80 dimensions total

    def __init__(self) -> None:
        rng = random.Random(42)
        self.attention_vector: List[float] = [rng.random() for _ in range(self.SEGMENT)]
        self.confidence_levels: List[float] = [0.5 + rng.gauss(0, 0.1) for _ in range(self.SEGMENT)]
        self.active_concepts: List[float] = [rng.gauss(0, 0.5) for _ in range(self.SEGMENT)]
        self.uncertainty_map: List[float] = [abs(rng.gauss(0.5, 0.2)) for _ in range(self.SEGMENT)]
        self.goal_state: List[float] = [rng.gauss(0, 1) for _ in range(self.SEGMENT)]
        # Full concatenated vector for scoring
        self._prev: Optional[List[float]] = None
        self._step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def vector(self) -> List[float]:
        """Full C(t) vector (concatenation of all sub-vectors)."""
        return (
            self.attention_vector
            + self.confidence_levels
            + self.active_concepts
            + self.uncertainty_map
            + self.goal_state
        )

    def update(
        self,
        external_input: List[float],
        memory_vector: Optional[List[float]] = None,
        learning_rate: float = 0.01,
    ) -> None:
        """
        Recursive update: C(t+1) = f(C(t), Input(t), Memory(t))

        A simple gated recurrent blend ensures stability while remaining
        sensitive to new information.
        """
        self._prev = self.vector[:]
        n = self.SEGMENT
        # Pad / truncate external_input to TOTAL_DIM
        inp = (external_input + [0.0] * self.TOTAL_DIM)[: self.TOTAL_DIM]
        mem = (memory_vector + [0.0] * self.TOTAL_DIM)[: self.TOTAL_DIM] if memory_vector else [0.0] * self.TOTAL_DIM

        combined = [i + 0.3 * m for i, m in zip(inp, mem)]
        alpha = learning_rate  # blend factor

        def _blend(current: List[float], offset: int) -> List[float]:
            return [
                (1 - alpha) * c + alpha * combined[offset + k]
                for k, c in enumerate(current)
            ]

        self.attention_vector = _blend(self.attention_vector, 0)
        self.confidence_levels = _blend(self.confidence_levels, n)
        self.active_concepts = _blend(self.active_concepts, 2 * n)
        self.uncertainty_map = _blend(self.uncertainty_map, 3 * n)
        self.goal_state = _blend(self.goal_state, 4 * n)
        self._step += 1

    def apply_error_feedback(self, error: float, learning_rate: float = 0.01) -> None:
        """
        Self-evaluation update:
          confidence = confidence - learning_rate * error
        """
        self.confidence_levels = [
            max(0.0, min(1.0, c - learning_rate * error))
            for c in self.confidence_levels
        ]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """H(C_t) = -Σ p_i log p_i  (uncertainty measure)."""
        return _entropy(self.vector)

    def stability(self) -> float:
        """||C_t - C_{t-1}||  (change from last step; 0 on first step)."""
        if self._prev is None:
            return 0.0
        diff = [a - b for a, b in zip(self.vector, self._prev)]
        return _norm(diff)

    def consistency(self, prediction_error: float) -> float:
        """1 - prediction_error  (clamped to [0, 1])."""
        return max(0.0, min(1.0, 1.0 - abs(prediction_error)))

    def metrics(self) -> Dict[str, float]:
        return {
            "entropy": self.entropy(),
            "stability": self.stability(),
            "step": float(self._step),
        }


# ---------------------------------------------------------------------------
# FSOT Thought Engine
# ---------------------------------------------------------------------------

class FSotThoughtEngine:
    """
    Manages a library of FSOT thoughts and selects them against C(t).

    score(thought_i) = C_t · W_i
    P(thought_i)     = softmax(score_i / temperature)
    W update:  W_thought += η * gradient(performance)
    """

    def __init__(self, temperature: float = 0.8, meta_lr: float = 0.05) -> None:
        self.temperature = temperature
        self.meta_lr = meta_lr
        self.thoughts: List[Thought] = self._default_thoughts()
        # Long-term memory: list of {state, action, result}
        self.memory: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Default thought library for nonce generation
    # ------------------------------------------------------------------

    @staticmethod
    def _default_thoughts() -> List[Thought]:
        return [
            Thought("if nonce increases by 1 → try sequential", 0.6, "nonce", "init"),
            Thought("if hash prefix has many zeros → exploit vicinity", 0.7, "hash", "init"),
            Thought("if error large → random jump in nonce space", 0.5, "nonce", "init"),
            Thought("if confidence low → widen search radius", 0.45, "numeric", "init"),
            Thought("if pattern repeats → flip high bits of nonce", 0.55, "nonce", "init"),
            Thought("if entropy high → reduce temperature", 0.6, "meta", "init"),
            Thought("if stability low → anchor to recent good nonce", 0.5, "nonce", "init"),
            Thought("if goal similarity high → exploit current direction", 0.65, "hash", "init"),
            Thought("if hash suffix matches target → neighbor search", 0.7, "hash", "init"),
            Thought("always maintain diversity in nonce candidates", 0.5, "numeric", "init"),
        ]

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def score_thoughts(self, consciousness: ConsciousnessState) -> List[float]:
        """Compute dot-product scores for all thoughts against C(t)."""
        c = consciousness.vector
        # The consciousness vector has TOTAL_DIM=80; thoughts embed to DIM=64.
        # We project c down to Thought.DIM by summing pairs.
        step = len(c) // Thought.DIM
        if step < 1:
            step = 1
        c_proj = [
            sum(c[i * step: (i + 1) * step]) / step
            for i in range(Thought.DIM)
        ]
        return [_dot(c_proj, t.embedding) for t in self.thoughts]

    def select_thought(
        self,
        consciousness: ConsciousnessState,
        thought_bias: Optional[Dict[str, float]] = None,
    ) -> Tuple[Thought, float]:
        """
        Sample a thought according to P(thought_i) = softmax(score_i / T).

        When ``thought_bias`` is provided, it maps thought patterns to additive
        bias scores so memory-guided strategies can reinforce or dampen specific
        thoughts before sampling.

        Returns (selected_thought, probability).
        """
        scores = self.score_thoughts(consciousness)
        if thought_bias:
            scores = [
                score + thought_bias.get(thought.pattern, 0.0)
                for thought, score in zip(self.thoughts, scores)
            ]
        probs = _softmax(scores, self.temperature)
        r = random.random()
        cumulative = 0.0
        for thought, prob in zip(self.thoughts, probs):
            cumulative += prob
            if r <= cumulative:
                return thought, prob
        return self.thoughts[-1], probs[-1]

    def generate_nonce_candidates(
        self, consciousness: ConsciousnessState, count: int = 8
    ) -> List[int]:
        """
        Use consciousness-guided thought selection to generate nonce candidates.
        Instead of pure random nonce generation, the consciousness state steers
        the search strategy.

        Returns a list of nonce integers (uint32 range).
        """
        candidates: List[int] = []
        for _ in range(count):
            thought, _prob = self.select_thought(consciousness)
            nonce = self._thought_to_nonce(thought, consciousness)
            candidates.append(nonce & 0xFFFFFFFF)
        return candidates

    def _thought_to_nonce(
        self, thought: Thought, consciousness: ConsciousnessState
    ) -> int:
        """
        Decode a nonce from a thought + current consciousness state.

        The consciousness vector is mixed with the thought's embedding and
        compressed to a 32-bit integer. This is the "decode(selected)" step
        from the architecture spec.
        """
        # Mix consciousness goal state with thought embedding (first 16 dims)
        goal = consciousness.goal_state
        seg = ConsciousnessState.SEGMENT
        mixed = [g * e for g, e in zip(goal, (thought.embedding[:seg]))]
        # Fold mixed vector into a bytes sequence, then interpret as uint32
        packed = struct.pack(
            ">" + "f" * len(mixed), *mixed
        )
        h = hashlib.sha256(packed).digest()
        nonce = struct.unpack(">I", h[:4])[0]
        return nonce

    def meta_learn(self, thought: Thought, performance_delta: float) -> None:
        """
        W_thought += η * gradient(performance)
        Adjust both the embedding and the confidence of a thought.
        """
        thought.confidence = max(0.0, min(1.0, thought.confidence + self.meta_lr * performance_delta))
        thought.embedding = [
            w + self.meta_lr * performance_delta * random.gauss(0, 0.1)
            for w in thought.embedding
        ]

    def remember(self, state_vec: List[float], thought: Thought, result: Any) -> None:
        """Store experience in long-term memory."""
        self.memory.append({"state": state_vec[:], "action": thought.to_dict(), "result": result})
        # Keep memory bounded
        if len(self.memory) > 1000:
            self.memory = self.memory[-1000:]

    def retrieve_similar(self, consciousness: ConsciousnessState, top_k: int = 3) -> List[Dict]:
        """Retrieve top-k most similar past experiences."""
        if not self.memory:
            return []
        c = consciousness.vector
        scored = [
            (_dot(c, m["state"][: len(c)]), m)
            for m in self.memory
            if len(m["state"]) >= len(c)
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    def memory_guidance(
        self, consciousness: ConsciousnessState, top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Convert similar past experiences into a memory vector and thought bias.

        Correct past predictions reinforce similar future states and thoughts,
        while incorrect past predictions dampen repeated mistakes.
        """
        similar = self.retrieve_similar(consciousness, top_k=top_k)
        dim = len(consciousness.vector)
        if not similar:
            return {
                "memory_vector": [0.0] * dim,
                "thought_bias": {},
                "sample_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "preferred_pattern": "",
            }

        memory_vector = [0.0] * dim
        thought_bias: Dict[str, float] = {}
        success_count = 0
        failure_count = 0

        for rank, item in enumerate(similar):
            state = list(item.get("state", []))
            action = item.get("action", {})
            result = item.get("result", {})
            reward = result.get("reward")
            if reward is None:
                reward = 1.0 if result.get("correct") else -1.0
            reward = max(-1.0, min(1.0, float(reward)))
            if reward >= 0.0:
                success_count += 1
            else:
                failure_count += 1

            pattern = str(action.get("pattern", ""))
            confidence = result.get("thought_confidence", action.get("confidence", 0.5))
            confidence = max(0.1, min(1.0, float(confidence)))
            recency = 1.0 / float(rank + 1)
            weight = reward * confidence * recency

            for index, value in enumerate(state[:dim]):
                memory_vector[index] += value * weight
            if pattern:
                thought_bias[pattern] = thought_bias.get(pattern, 0.0) + weight

        peak = max((abs(value) for value in memory_vector), default=0.0)
        if peak > 0.0:
            memory_vector = [value / peak for value in memory_vector]

        preferred_pattern = ""
        if thought_bias:
            pattern, score = max(thought_bias.items(), key=lambda item: item[1])
            if score > 0.0:
                preferred_pattern = pattern

        return {
            "memory_vector": memory_vector,
            "thought_bias": thought_bias,
            "sample_count": len(similar),
            "success_count": success_count,
            "failure_count": failure_count,
            "preferred_pattern": preferred_pattern,
        }

    @property
    def SEGMENT(self) -> int:  # pragma: no cover
        return ConsciousnessState.SEGMENT


# ---------------------------------------------------------------------------
# Evaluate mining hash quality
# ---------------------------------------------------------------------------

def evaluate_hash_quality(hash_hex: str, target_bits: int = 32) -> float:
    """
    Returns an error signal in [0, 1] based on how many leading zero bits
    the hash has relative to the target.

    error = 0   → hash meets or exceeds the target difficulty
    error = 1   → hash has zero leading zeros
    """
    try:
        hash_int = int(hash_hex, 16)
    except ValueError:
        return 1.0
    if hash_int == 0:
        return 0.0
    leading_zeros = 256 - hash_int.bit_length()
    ratio = leading_zeros / max(target_bits, 1)
    error = max(0.0, 1.0 - ratio)
    return error


def sha256d(data: bytes) -> bytes:
    """Double SHA-256 (standard Bitcoin / SHA-256 Stratum hashing)."""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()


def build_block_header(
    version: str,
    prevhash: str,
    merkle_root_bytes: bytes,
    ntime: str,
    nbits: str,
    nonce: int,
) -> bytes:
    """
    Construct the 80-byte Bitcoin block header for SHA-256 mining.
    All hex strings are treated as little-endian as per the Stratum protocol.
    """
    version_bytes = struct.pack("<I", int(version, 16))
    prevhash_bytes = binascii.unhexlify(prevhash)[::-1]
    ntime_bytes = struct.pack("<I", int(ntime, 16))
    nbits_bytes = struct.pack("<I", int(nbits, 16))
    nonce_bytes = struct.pack("<I", nonce)
    return version_bytes + prevhash_bytes + merkle_root_bytes + ntime_bytes + nbits_bytes + nonce_bytes
