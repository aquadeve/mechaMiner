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
