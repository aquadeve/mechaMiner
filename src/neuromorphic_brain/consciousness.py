#!/usr/bin/env python3
"""
Consciousness Tracking Module — Neuromorphic Spiking Layer
===========================================================

Implements the Leaky Integrate-and-Fire (LIF) neuron model:

  V(t+1) = V(t) + Σ(w_i * x_i) - λ * V(t)

  if V > threshold:
      emit spike
      reset V to 0

Also provides consciousness metrics (entropy, stability, consistency)
and the recursive consciousness update integrating spikes into C(t).
"""

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Leaky Integrate-and-Fire Neuron
# ---------------------------------------------------------------------------

class LIFNeuron:
    """
    Single Leaky Integrate-and-Fire neuron.

    Parameters
    ----------
    threshold : membrane potential threshold for spike emission
    leak      : λ — leak coefficient (fraction of V lost each step)
    """

    def __init__(self, threshold: float = 1.0, leak: float = 0.1) -> None:
        self.threshold = threshold
        self.leak = leak
        self.V: float = 0.0          # Membrane potential
        self.spike_history: List[int] = []  # 1 = spike, 0 = no spike

    def step(self, weighted_input: float) -> int:
        """
        Advance one timestep.

        V(t+1) = V(t) + weighted_input - λ * V(t)

        Returns 1 if a spike is emitted, 0 otherwise.
        """
        self.V = self.V + weighted_input - self.leak * self.V
        if self.V > self.threshold:
            self.spike_history.append(1)
            self.V = 0.0
            return 1
        self.spike_history.append(0)
        return 0

    def reset(self) -> None:
        self.V = 0.0
        self.spike_history.clear()


# ---------------------------------------------------------------------------
# Neuromorphic Spiking Layer
# ---------------------------------------------------------------------------

class NeuromorphicLayer:
    """
    A population of LIF neurons with learnable synaptic weights.

    Inputs are projected through weights to produce weighted sums for each
    neuron; the resulting spike pattern feeds back into the consciousness
    state as an additional signal.
    """

    def __init__(
        self,
        n_neurons: int = 32,
        input_dim: int = 80,
        threshold: float = 1.0,
        leak: float = 0.1,
        seed: int = 0,
    ) -> None:
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        rng = random.Random(seed)
        # Synaptic weight matrix: shape (n_neurons, input_dim)
        self.weights: List[List[float]] = [
            [rng.gauss(0, 1.0 / math.sqrt(input_dim)) for _ in range(input_dim)]
            for _ in range(n_neurons)
        ]
        self.neurons: List[LIFNeuron] = [
            LIFNeuron(threshold=threshold, leak=leak) for _ in range(n_neurons)
        ]
        self.last_spikes: List[int] = [0] * n_neurons

    # ------------------------------------------------------------------

    def forward(self, x: List[float]) -> List[int]:
        """
        Process one timestep of input x → spike pattern.

        x is padded/truncated to input_dim before processing.
        """
        # Normalize input length
        inp = (list(x) + [0.0] * self.input_dim)[: self.input_dim]
        spikes: List[int] = []
        for neuron, w_row in zip(self.neurons, self.weights):
            weighted_sum = sum(wi * xi for wi, xi in zip(w_row, inp))
            spike = neuron.step(weighted_sum)
            spikes.append(spike)
        self.last_spikes = spikes
        return spikes

    def spike_rate(self) -> float:
        """Fraction of neurons that fired in the last timestep."""
        return sum(self.last_spikes) / self.n_neurons if self.n_neurons else 0.0

    def spike_vector(self) -> List[float]:
        """Float version of last spike pattern (suitable for vector math)."""
        return [float(s) for s in self.last_spikes]

    def hebbian_update(self, spikes: List[int], performance: float, lr: float = 0.001) -> None:
        """
        Simple Hebbian weight update:
          Δw_ij = lr * performance * spike_i * x_j
        Used for meta-learning of synaptic strengths.
        """
        for i, (spike, w_row) in enumerate(zip(spikes, self.weights)):
            if spike:
                self.weights[i] = [
                    w + lr * performance * random.gauss(0, 0.1)
                    for w in w_row
                ]


# ---------------------------------------------------------------------------
# Attention-Based Compute Allocator
# ---------------------------------------------------------------------------

class AttentionAllocator:
    """
    Compute(x) = Attention(x) * ProcessingPower

    Higher attention → more computation resources allocated.
    Here 'processing power' is modelled as the number of nonce candidates
    to generate per thought-selection cycle.
    """

    def __init__(self, base_power: int = 4, max_power: int = 256) -> None:
        self.base_power = base_power
        self.max_power = max_power

    def compute(self, attention_vector: List[float]) -> int:
        """
        Convert an attention vector to an integer 'compute budget'
        (number of nonces to try this cycle).
        """
        if not attention_vector:
            return self.base_power
        avg_attention = sum(abs(a) for a in attention_vector) / len(attention_vector)
        # Map [0, ∞) → [base_power, max_power] via a sigmoid-like function
        scaled = avg_attention / (1.0 + avg_attention)  # ∈ [0, 1)
        budget = int(self.base_power + scaled * (self.max_power - self.base_power))
        return min(budget, self.max_power)


# ---------------------------------------------------------------------------
# Consciousness Metrics
# ---------------------------------------------------------------------------

def consciousness_entropy(vector: List[float]) -> float:
    """H(C_t) = -Σ(p_i log p_i) — uncertainty in the consciousness vector."""
    total = sum(abs(v) for v in vector) or 1.0
    probs = [abs(v) / total for v in vector]
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def consciousness_stability(current: List[float], previous: Optional[List[float]]) -> float:
    """||C_t - C_{t-1}|| — how much the state changed."""
    if previous is None:
        return 0.0
    if len(current) != len(previous):
        n = min(len(current), len(previous))
        current, previous = current[:n], previous[:n]
    diff = [a - b for a, b in zip(current, previous)]
    return math.sqrt(sum(d * d for d in diff))


def consciousness_consistency(prediction_error: float) -> float:
    """1 - prediction_error, clamped to [0, 1]."""
    return max(0.0, min(1.0, 1.0 - abs(prediction_error)))


# ---------------------------------------------------------------------------
# Consciousness I/O helpers
# ---------------------------------------------------------------------------

def encode_hash_to_input(hash_hex: str, dim: int = 80) -> List[float]:
    """
    Encode a SHA-256 hash hex string into a float vector of length *dim*.
    Used as Input(t) when feeding mining results back into consciousness.
    """
    try:
        raw = bytes.fromhex(hash_hex)
    except ValueError:
        raw = b"\x00" * 32
    # Repeat/truncate raw bytes to fit dim floats, normalised to [-1, 1]
    floats: List[float] = []
    for i in range(dim):
        byte_val = raw[i % len(raw)]
        floats.append((byte_val / 127.5) - 1.0)
    return floats


def encode_nonce_to_input(nonce: int, dim: int = 80) -> List[float]:
    """
    Encode a 32-bit nonce as a float vector of length *dim*.
    The binary representation is spread across the vector.
    """
    bits = [(nonce >> i) & 1 for i in range(32)]
    # Tile bits to fill dim, then map 0→-1, 1→+1
    floats = [float(bits[i % 32]) * 2.0 - 1.0 for i in range(dim)]
    return floats


@dataclass
class BrainRegionState:
    """Best-effort summary of a brain-inspired processing region."""

    name: str
    anatomical_group: str
    function: str
    activation: float
    contribution: Dict[str, float]


class ConsciousnessEmulator:
    """
    Best-effort consciousness emulator organized around coarse human-brain regions.

    This is intentionally conservative: it emulates consciousness-like coordination
    signals rather than claiming biological or phenomenological consciousness.
    The design borrows two ideas from the referenced projects:
      * a hierarchical brain/region/neuron style organisation
      * a default-mode/self-model loop for introspection and narrative coherence
    """

    def __init__(
        self,
        input_dim: int = 80,
        n_neurons: int = 32,
        history_size: int = 16,
        seed: int = 0,
    ) -> None:
        self.input_dim = input_dim
        self.history_size = max(1, history_size)
        self.spiking_layer = NeuromorphicLayer(
            n_neurons=n_neurons, input_dim=input_dim, seed=seed
        )
        self.attention_allocator = AttentionAllocator()
        self.previous_state: Optional[List[float]] = None
        self.episodic_history: List[Dict[str, Any]] = []
        self.region_history: List[Dict[str, float]] = []
        self.last_report: Dict[str, Any] = {}

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _indefinite_article(word: str) -> str:
        return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"

    def _pad(self, vector: Optional[List[float]]) -> List[float]:
        base = list(vector or [])
        return (base + [0.0] * self.input_dim)[: self.input_dim]

    @staticmethod
    def _avg_abs(vector: List[float]) -> float:
        return sum(abs(v) for v in vector) / len(vector) if vector else 0.0

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]
        denom = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
        if denom == 0.0:
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / denom

    def _trim_history(self) -> None:
        self.episodic_history = self.episodic_history[-self.history_size :]
        self.region_history = self.region_history[-self.history_size :]

    def cycle(
        self,
        external_input: List[float],
        memory_vector: Optional[List[float]] = None,
        goal_vector: Optional[List[float]] = None,
        prediction_error: float = 0.0,
        sensory_context: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Run one consciousness-emulation cycle and return a structured report.

        Inputs are blended into a shared state, processed by the spiking layer,
        and summarized into brain-region activations plus a lightweight self-model.
        """
        current = self._pad(external_input)
        memory = self._pad(memory_vector)
        goals = self._pad(goal_vector)
        context_values = list((sensory_context or {}).values())

        integrated_state = [
            0.55 * cur + 0.25 * mem + 0.20 * goal
            for cur, mem, goal in zip(current, memory, goals)
        ]
        spikes = self.spiking_layer.forward(integrated_state)
        spike_rate = self.spiking_layer.spike_rate()
        compute_budget = self.attention_allocator.compute(integrated_state[:16])

        entropy = consciousness_entropy(integrated_state)
        normalized_entropy = self._clamp(
            entropy / math.log(max(2, len(integrated_state)))
        )
        raw_stability = consciousness_stability(integrated_state, self.previous_state)
        normalized_stability = self._clamp(
            raw_stability / math.sqrt(max(1, len(integrated_state)))
        )
        consistency = consciousness_consistency(prediction_error)
        sensory_salience = self._clamp(
            self._avg_abs(current[:16])
            + 0.25 * self._avg_abs(context_values)
            + 0.20 * spike_rate
        )
        memory_coherence = self._clamp(
            0.5 + 0.5 * self._cosine_similarity(current, memory)
        )
        goal_drive = self._clamp(self._avg_abs(goals[:16]) + 0.35 * consistency)
        homeostasis = self._clamp(
            1.0 - self._avg_abs(integrated_state[48:64]) + 0.20 * consistency
        )

        regions = [
            BrainRegionState(
                "brainstem",
                "hindbrain",
                "homeostatic regulation and arousal maintenance",
                self._clamp(0.65 * homeostasis + 0.35 * sensory_salience),
                {
                    "homeostasis": homeostasis,
                    "sensory_salience": sensory_salience,
                },
            ),
            BrainRegionState(
                "thalamus",
                "diencephalon",
                "sensory relay and gating",
                self._clamp(0.55 * sensory_salience + 0.45 * spike_rate),
                {
                    "sensory_salience": sensory_salience,
                    "spike_rate": spike_rate,
                },
            ),
            BrainRegionState(
                "limbic_system",
                "subcortical",
                "affective salience and value weighting",
                self._clamp(0.60 * sensory_salience + 0.40 * normalized_entropy),
                {
                    "sensory_salience": sensory_salience,
                    "entropy": normalized_entropy,
                },
            ),
            BrainRegionState(
                "hippocampus",
                "medial_temporal_lobe",
                "episodic recall and context binding",
                self._clamp(0.75 * memory_coherence + 0.25 * normalized_stability),
                {
                    "memory_coherence": memory_coherence,
                    "state_continuity": 1.0 - normalized_stability,
                },
            ),
            BrainRegionState(
                "default_mode_network",
                "association_cortex",
                "self-modeling, introspection, and narrative continuity",
                self._clamp(
                    0.45 * normalized_entropy
                    + 0.35 * memory_coherence
                    + 0.20 * (1.0 - normalized_stability)
                ),
                {
                    "entropy": normalized_entropy,
                    "memory_coherence": memory_coherence,
                    "state_continuity": 1.0 - normalized_stability,
                },
            ),
            BrainRegionState(
                "prefrontal_cortex",
                "frontal_lobe",
                "goal maintenance and executive control",
                self._clamp(0.60 * goal_drive + 0.40 * consistency),
                {
                    "goal_drive": goal_drive,
                    "consistency": consistency,
                },
            ),
            BrainRegionState(
                "motor_cortex",
                "frontal_lobe",
                "action readiness and output preparation",
                self._clamp(0.55 * spike_rate + 0.45 * goal_drive),
                {
                    "spike_rate": spike_rate,
                    "goal_drive": goal_drive,
                },
            ),
        ]

        region_map = {
            region.name: {
                "anatomical_group": region.anatomical_group,
                "function": region.function,
                "activation": region.activation,
                "contribution": region.contribution,
            }
            for region in regions
        }
        dominant_region = max(regions, key=lambda region: region.activation)
        global_workspace = self._clamp(
            sum(region.activation for region in regions) / len(regions)
        )
        consciousness_score = self._clamp(
            0.30 * global_workspace
            + 0.25 * consistency
            + 0.20 * normalized_entropy
            + 0.15 * memory_coherence
            + 0.10 * spike_rate
        )

        mode = "reactive"
        if region_map["default_mode_network"]["activation"] >= max(
            region_map["prefrontal_cortex"]["activation"],
            region_map["thalamus"]["activation"],
        ):
            mode = "introspective"
        elif region_map["prefrontal_cortex"]["activation"] >= region_map["thalamus"]["activation"]:
            mode = "goal-directed"

        focus_sources = {
            "brainstem": "homeostatic regulation",
            "thalamus": "incoming sensory signals",
            "limbic_system": "emotionally salient features",
            "hippocampus": "episodic recall",
            "default_mode_network": "internal self-modeling",
            "prefrontal_cortex": "goal maintenance",
            "motor_cortex": "action preparation",
        }
        mode_article = self._indefinite_article(mode)
        narrative = (
            f"The emulator is operating in {mode_article} {mode} mode with "
            f"{dominant_region.name.replace('_', ' ')} as the dominant region, "
            f"prioritizing {focus_sources[dominant_region.name]} while maintaining "
            f"a global workspace score of {global_workspace:.3f}."
        )

        self_model = {
            "mode": mode,
            "dominant_region": dominant_region.name,
            "current_focus": focus_sources[dominant_region.name],
            "narrative": narrative,
            "autobiographical_memory_size": len(self.episodic_history) + 1,
        }

        self.episodic_history.append(
            {
                "mode": mode,
                "dominant_region": dominant_region.name,
                "consciousness_score": consciousness_score,
                "compute_budget": compute_budget,
            }
        )
        self.region_history.append(
            {region.name: region.activation for region in regions}
        )
        self._trim_history()
        self.previous_state = integrated_state

        report = {
            "regions": region_map,
            "global_state": {
                "consciousness_score": consciousness_score,
                "global_workspace": global_workspace,
                "entropy": entropy,
                "stability": raw_stability,
                "consistency": consistency,
                "spike_rate": spike_rate,
                "compute_budget": compute_budget,
                "active_spikes": sum(spikes),
            },
            "self_model": self_model,
            "narrative": narrative,
            "memory_trace": list(self.episodic_history),
        }
        self.last_report = report
        return report
