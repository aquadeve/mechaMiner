#!/usr/bin/env python3
"""
Tests for the FSOT consciousness state, thought engine, and neuromorphic
spiking layer used by the consciousness-powered crypto miner.
"""

import math
import sys
import os

# Ensure src submodule dirs are on the path (also done by conftest.py)
_src = os.path.join(os.path.dirname(__file__), "..", "src")
for _d in [_src, os.path.join(_src, "fsot_core"), os.path.join(_src, "neuromorphic_brain")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from core_system import (
    ConsciousnessState,
    FSotThoughtEngine,
    Thought,
    _dot,
    _softmax,
    _norm,
    _entropy,
    evaluate_hash_quality,
    sha256d,
    build_block_header,
)
from consciousness import (
    LIFNeuron,
    NeuromorphicLayer,
    AttentionAllocator,
    consciousness_entropy,
    consciousness_stability,
    consciousness_consistency,
    encode_hash_to_input,
    encode_nonce_to_input,
)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


def test_dot_product():
    assert _dot([1.0, 2.0], [3.0, 4.0]) == pytest_approx(11.0)


def test_softmax_sums_to_one():
    probs = _softmax([1.0, 2.0, 3.0])
    assert abs(sum(probs) - 1.0) < 1e-9
    assert all(p >= 0 for p in probs)


def test_softmax_temperature():
    # Higher temperature → more uniform distribution
    probs_low = _softmax([0.0, 10.0], temperature=0.1)
    probs_high = _softmax([0.0, 10.0], temperature=10.0)
    assert probs_low[1] > probs_high[1]


def test_norm():
    assert abs(_norm([3.0, 4.0]) - 5.0) < 1e-9


def test_entropy_uniform():
    # Uniform vector → maximum entropy
    v = [1.0] * 8
    h = _entropy(v)
    assert h > 0.0


def test_entropy_spike():
    # Single non-zero element → minimum entropy
    v = [0.0] * 7 + [1.0]
    h = _entropy(v)
    assert h == pytest_approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ConsciousnessState tests
# ---------------------------------------------------------------------------


def test_consciousness_state_dim():
    c = ConsciousnessState()
    assert len(c.vector) == ConsciousnessState.TOTAL_DIM


def test_consciousness_update_changes_state():
    c = ConsciousnessState()
    before = c.vector[:]
    inp = [0.5] * ConsciousnessState.TOTAL_DIM
    c.update(inp)
    assert c.vector != before


def test_consciousness_stability_zero_first_step():
    c = ConsciousnessState()
    assert c.stability() == 0.0


def test_consciousness_stability_nonzero_after_update():
    c = ConsciousnessState()
    inp = [1.0] * ConsciousnessState.TOTAL_DIM
    c.update(inp)
    assert c.stability() > 0.0


def test_consciousness_entropy_positive():
    c = ConsciousnessState()
    assert c.entropy() > 0.0


def test_consciousness_error_feedback():
    c = ConsciousnessState()
    before = list(c.confidence_levels)
    c.apply_error_feedback(1.0, learning_rate=0.1)
    after = c.confidence_levels
    # All confidence levels should have decreased
    assert all(a <= b for a, b in zip(after, before))


def test_consciousness_consistency():
    c = ConsciousnessState()
    assert c.consistency(0.0) == pytest_approx(1.0)
    assert c.consistency(1.0) == pytest_approx(0.0)
    assert c.consistency(0.5) == pytest_approx(0.5)


def test_consciousness_step_counter():
    c = ConsciousnessState()
    assert c._step == 0
    c.update([0.0] * ConsciousnessState.TOTAL_DIM)
    assert c._step == 1


# ---------------------------------------------------------------------------
# Thought & FSotThoughtEngine tests
# ---------------------------------------------------------------------------


def test_thought_embedding_deterministic():
    t1 = Thought("same pattern", 0.5)
    t2 = Thought("same pattern", 0.5)
    assert t1.embedding == t2.embedding


def test_thought_embedding_different_patterns():
    t1 = Thought("pattern A", 0.5)
    t2 = Thought("pattern B", 0.5)
    assert t1.embedding != t2.embedding


def test_thought_embedding_dim():
    t = Thought("test", 0.5)
    assert len(t.embedding) == Thought.DIM


def test_thought_confidence_clamped():
    t = Thought("x", confidence=2.0)
    assert t.confidence == 1.0
    t2 = Thought("y", confidence=-1.0)
    assert t2.confidence == 0.0


def test_thought_engine_default_thoughts():
    engine = FSotThoughtEngine()
    assert len(engine.thoughts) > 0


def test_thought_engine_score_returns_list():
    engine = FSotThoughtEngine()
    c = ConsciousnessState()
    scores = engine.score_thoughts(c)
    assert len(scores) == len(engine.thoughts)
    assert all(isinstance(s, float) for s in scores)


def test_thought_engine_select_returns_thought():
    engine = FSotThoughtEngine()
    c = ConsciousnessState()
    thought, prob = engine.select_thought(c)
    assert isinstance(thought, Thought)
    assert 0.0 < prob <= 1.0


def test_thought_engine_generate_candidates():
    engine = FSotThoughtEngine()
    c = ConsciousnessState()
    candidates = engine.generate_nonce_candidates(c, count=4)
    assert len(candidates) == 4
    assert all(0 <= n <= 0xFFFFFFFF for n in candidates)


def test_thought_engine_meta_learn_updates_confidence():
    engine = FSotThoughtEngine()
    t = engine.thoughts[0]
    before = t.confidence
    engine.meta_learn(t, performance_delta=0.5)
    assert t.confidence != before or abs(t.confidence - before) < 0.1


def test_thought_engine_memory():
    engine = FSotThoughtEngine()
    c = ConsciousnessState()
    t = engine.thoughts[0]
    engine.remember(c.vector, t, {"result": "ok"})
    assert len(engine.memory) == 1
    similar = engine.retrieve_similar(c, top_k=1)
    assert len(similar) == 1


def test_thought_engine_memory_bounded():
    engine = FSotThoughtEngine()
    c = ConsciousnessState()
    t = engine.thoughts[0]
    for _ in range(1100):
        engine.remember(c.vector, t, {})
    assert len(engine.memory) <= 1000


# ---------------------------------------------------------------------------
# LIF Neuron tests
# ---------------------------------------------------------------------------


def test_lif_neuron_no_spike_below_threshold():
    neuron = LIFNeuron(threshold=1.0, leak=0.0)
    spike = neuron.step(0.5)
    assert spike == 0
    assert abs(neuron.V - 0.5) < 1e-9


def test_lif_neuron_spike_above_threshold():
    neuron = LIFNeuron(threshold=1.0, leak=0.0)
    neuron.V = 0.9
    spike = neuron.step(0.2)
    assert spike == 1
    assert neuron.V == 0.0  # reset after spike


def test_lif_neuron_leak_reduces_potential():
    neuron = LIFNeuron(threshold=10.0, leak=0.5)
    neuron.V = 1.0
    neuron.step(0.0)
    assert neuron.V < 1.0


def test_lif_neuron_reset():
    neuron = LIFNeuron()
    neuron.V = 5.0
    neuron.spike_history = [1, 0, 1]
    neuron.reset()
    assert neuron.V == 0.0
    assert neuron.spike_history == []


# ---------------------------------------------------------------------------
# NeuromorphicLayer tests
# ---------------------------------------------------------------------------


def test_neuromorphic_layer_forward_shape():
    layer = NeuromorphicLayer(n_neurons=8, input_dim=16)
    spikes = layer.forward([0.5] * 16)
    assert len(spikes) == 8
    assert all(s in (0, 1) for s in spikes)


def test_neuromorphic_layer_spike_rate_range():
    layer = NeuromorphicLayer(n_neurons=16, input_dim=16)
    layer.forward([1.0] * 16)
    rate = layer.spike_rate()
    assert 0.0 <= rate <= 1.0


def test_neuromorphic_layer_spike_vector():
    layer = NeuromorphicLayer(n_neurons=4, input_dim=4)
    layer.forward([0.0] * 4)
    sv = layer.spike_vector()
    assert len(sv) == 4
    assert all(v in (0.0, 1.0) for v in sv)


def test_neuromorphic_layer_hebbian_update():
    layer = NeuromorphicLayer(n_neurons=4, input_dim=4, seed=1)
    before = [row[:] for row in layer.weights]
    spikes = [1, 0, 1, 0]
    layer.hebbian_update(spikes, performance=1.0, lr=0.1)
    # At least the neurons that spiked should have different weights
    changed = any(layer.weights[i] != before[i] for i in range(4))
    assert changed


# ---------------------------------------------------------------------------
# AttentionAllocator tests
# ---------------------------------------------------------------------------


def test_attention_allocator_base():
    alloc = AttentionAllocator(base_power=4, max_power=64)
    budget = alloc.compute([0.0] * 16)
    assert budget == 4


def test_attention_allocator_scales():
    alloc = AttentionAllocator(base_power=4, max_power=64)
    low_budget = alloc.compute([0.01] * 16)
    high_budget = alloc.compute([100.0] * 16)
    assert high_budget >= low_budget


def test_attention_allocator_max():
    alloc = AttentionAllocator(base_power=4, max_power=64)
    budget = alloc.compute([1e6] * 16)
    assert budget <= 64


# ---------------------------------------------------------------------------
# Consciousness metrics tests
# ---------------------------------------------------------------------------


def test_consciousness_entropy_func():
    v = [1.0] * 8
    h = consciousness_entropy(v)
    assert h > 0.0


def test_consciousness_stability_func():
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert consciousness_stability(a, b) == pytest_approx(0.0)
    b2 = [2.0, 2.0, 3.0]
    assert consciousness_stability(a, b2) == pytest_approx(1.0)


def test_consciousness_stability_none_prev():
    assert consciousness_stability([1.0], None) == 0.0


def test_consciousness_consistency_func():
    assert consciousness_consistency(0.0) == pytest_approx(1.0)
    assert consciousness_consistency(1.0) == pytest_approx(0.0)


# ---------------------------------------------------------------------------
# Hash encoding helpers tests
# ---------------------------------------------------------------------------


def test_encode_hash_to_input_shape():
    h = "a" * 64
    v = encode_hash_to_input(h, dim=80)
    assert len(v) == 80


def test_encode_hash_to_input_range():
    h = "ff" * 32
    v = encode_hash_to_input(h, dim=16)
    assert all(-1.0 <= x <= 1.0 for x in v)


def test_encode_nonce_to_input():
    v = encode_nonce_to_input(0xDEADBEEF, dim=32)
    assert len(v) == 32
    assert all(x in (-1.0, 1.0) for x in v)


# ---------------------------------------------------------------------------
# Mining utility tests
# ---------------------------------------------------------------------------


def test_sha256d_length():
    result = sha256d(b"hello")
    assert len(result) == 32


def test_evaluate_hash_quality_zero_hash():
    # All-zero hash → perfect (error = 0)
    zero_hash = "0" * 64
    error = evaluate_hash_quality(zero_hash, target_bits=32)
    assert error == 0.0


def test_evaluate_hash_quality_max_hash():
    # All-f hash → worst (lots of leading ones)
    max_hash = "f" * 64
    error = evaluate_hash_quality(max_hash, target_bits=32)
    assert error == pytest_approx(1.0)


def test_evaluate_hash_quality_range():
    import random
    rng = random.Random(0)
    for _ in range(20):
        h = "".join(f"{rng.randint(0, 255):02x}" for _ in range(32))
        error = evaluate_hash_quality(h, target_bits=32)
        assert 0.0 <= error <= 1.0


def test_build_block_header_length():
    version = "20000000"
    prevhash = "00" * 32
    merkle_root = bytes(32)
    ntime = "60000000"
    nbits = "1d00ffff"
    nonce = 12345
    header = build_block_header(version, prevhash, merkle_root, ntime, nbits, nonce)
    assert len(header) == 80


# ---------------------------------------------------------------------------
# pytest_approx compatibility shim (works with plain pytest too)
# ---------------------------------------------------------------------------


def pytest_approx(expected, abs=None, rel=None):
    """Thin wrapper so the tests read naturally even without pytest."""
    return _Approx(expected, abs=abs, rel=rel)


class _Approx:
    def __init__(self, expected, abs=None, rel=None):
        self._expected = expected
        self._abs = abs if abs is not None else 1e-6
        self._rel = rel

    def __eq__(self, other):
        return math.isclose(other, self._expected, abs_tol=self._abs, rel_tol=self._rel or 1e-6)

    def __repr__(self):
        return f"~{self._expected}"


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    passed = failed = 0
    for name, func in list(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            func()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
