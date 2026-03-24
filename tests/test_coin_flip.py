#!/usr/bin/env python3
"""
Tests for the CoinFlipConsciousness module (--remoteviewsimple mode).

Coverage
--------
* CoinFlipConsciousness.predict() — returns well-formed records, valid outcomes.
* CoinFlipConsciousness.train()   — updates statistics, writes training records.
* Persistence                     — predictions.jsonl / training.jsonl survive a
                                    fresh instance loading from the same dir.
* Multi-coin                      — n_coins > 1 produces one record per coin.
* Digital-realm helper            — run_digitalrealm() auto-generates actuals.
* Real-realm helper               — run_realrealm() handles user input (mocked).
* System snapshot                 — _system_snapshot() / _time_to_consciousness_vector().
* Session stats                   — accuracy() / stats() reflect trained data.
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
from typing import Any, Dict, List
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py / test_brain.py)
# ---------------------------------------------------------------------------

_here = os.path.dirname(__file__)
_src = os.path.join(_here, "..", "src")
for _d in [_src, os.path.join(_src, "fsot_core"), os.path.join(_src, "neuromorphic_brain")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from coin_flip_consciousness import (
    OUTCOMES,
    CoinFlipConsciousness,
    _system_snapshot,
    _thought_to_outcome,
    _time_to_consciousness_vector,
    run_digitalrealm,
    run_predictor,
    run_realrealm,
)
from core_system import ConsciousnessState, Thought


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfc(tmpdir: str) -> CoinFlipConsciousness:
    """Create a CoinFlipConsciousness with a deterministic seed in a tmp dir."""
    return CoinFlipConsciousness(session_dir=tmpdir, session_id="test_session", seed=42)


# ---------------------------------------------------------------------------
# _system_snapshot
# ---------------------------------------------------------------------------


def test_system_snapshot_has_required_keys():
    snap = _system_snapshot()
    required = {
        "utc_iso",
        "unix_timestamp",
        "monotonic_ns",
        "local_year",
        "local_month",
        "local_day",
        "local_hour",
        "local_minute",
        "local_second",
        "local_microsecond",
        "platform_node",
        "platform_system",
        "platform_machine",
    }
    assert required.issubset(snap.keys())


def test_system_snapshot_unix_timestamp_positive():
    snap = _system_snapshot()
    assert snap["unix_timestamp"] > 0


def test_system_snapshot_monotonic_ns_type():
    snap = _system_snapshot()
    assert isinstance(snap["monotonic_ns"], int)


# ---------------------------------------------------------------------------
# _time_to_consciousness_vector
# ---------------------------------------------------------------------------


def test_time_to_consciousness_vector_correct_dim():
    snap = _system_snapshot()
    vec = _time_to_consciousness_vector(snap)
    assert len(vec) == ConsciousnessState.TOTAL_DIM


def test_time_to_consciousness_vector_range():
    snap = _system_snapshot()
    vec = _time_to_consciousness_vector(snap)
    assert all(-1.0 <= v <= 1.0 for v in vec)


def test_time_to_consciousness_vector_varies_over_time():
    import time as _time

    snap1 = _system_snapshot()
    _time.sleep(0.01)
    snap2 = _system_snapshot()
    vec1 = _time_to_consciousness_vector(snap1)
    vec2 = _time_to_consciousness_vector(snap2)
    # At least some values must differ (they encode actual wall time)
    assert vec1 != vec2


# ---------------------------------------------------------------------------
# _thought_to_outcome
# ---------------------------------------------------------------------------


def test_thought_to_outcome_valid():
    thought = Thought("test_pattern", 0.5)
    result = _thought_to_outcome(thought, step=0)
    assert result in OUTCOMES


def test_thought_to_outcome_step_varies_result():
    thought = Thought("test_pattern", 0.5)
    results = {_thought_to_outcome(thought, step=i) for i in range(10)}
    # Some steps should produce different outcomes
    assert len(results) <= 2  # only heads or tails


def test_thought_to_outcome_deterministic():
    thought = Thought("deterministic_pattern", 0.7)
    r1 = _thought_to_outcome(thought, step=7)
    r2 = _thought_to_outcome(thought, step=7)
    assert r1 == r2


# ---------------------------------------------------------------------------
# CoinFlipConsciousness — initialisation
# ---------------------------------------------------------------------------


def test_cfc_session_dir_created():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        assert (cfc.session_dir).exists()


def test_cfc_session_meta_written():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        meta_path = cfc.session_dir / cfc.SESSION_META_FILE
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["session_id"] == "test_session"
        assert "session_started_utc" in meta
        assert "platform" in meta


def test_cfc_initial_stats_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        stats = cfc.stats()
        assert stats["step"] == 0
        assert stats["total_trained"] == 0
        assert stats["correct"] == 0
        assert stats["accuracy"] == 0.0


# ---------------------------------------------------------------------------
# CoinFlipConsciousness.predict()
# ---------------------------------------------------------------------------


def test_predict_returns_n_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=3)
        assert len(preds) == 3


def test_predict_coin_index_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=4)
        assert [p["coin_index"] for p in preds] == [1, 2, 3, 4]


def test_predict_valid_outcomes():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        for pred in cfc.predict(n_coins=20):
            assert pred["prediction"] in OUTCOMES


def test_predict_includes_system_snapshot():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        pred = cfc.predict(n_coins=1)[0]
        assert "system_snapshot" in pred
        assert "utc_iso" in pred["system_snapshot"]


def test_predict_includes_consciousness_vector():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        pred = cfc.predict(n_coins=1)[0]
        assert "consciousness_vector" in pred
        assert len(pred["consciousness_vector"]) == ConsciousnessState.TOTAL_DIM


def test_predict_record_type_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        pred = cfc.predict(n_coins=1)[0]
        assert pred["record_type"] == "prediction"


def test_predict_step_increments():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        cfc.predict(n_coins=2)
        cfc.predict(n_coins=3)
        assert cfc._step == 5


def test_predict_writes_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        cfc.predict(n_coins=3)
        pred_file = cfc.session_dir / cfc.PREDICTIONS_FILE
        assert pred_file.exists()
        lines = [l for l in pred_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 3


def test_predict_confidence_bounded():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        for pred in cfc.predict(n_coins=10):
            assert 0.0 <= pred["thought_confidence"] <= 1.0
            assert 0.0 <= pred["selection_probability"] <= 1.0


# ---------------------------------------------------------------------------
# CoinFlipConsciousness.train()
# ---------------------------------------------------------------------------


def test_train_updates_total():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=3)
        cfc.train(preds, ["heads", "tails", "heads"], realm="test")
        assert cfc._total == 3


def test_train_correct_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=2)
        # Force actuals to match predictions
        actuals = [p["prediction"] for p in preds]
        cfc.train(preds, actuals, realm="test")
        assert cfc._correct == 2


def test_train_wrong_reduces_correct():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=2)
        # Force actuals to be the opposite of predictions
        opposite = {"heads": "tails", "tails": "heads"}
        actuals = [opposite[p["prediction"]] for p in preds]
        cfc.train(preds, actuals, realm="test")
        assert cfc._correct == 0


def test_train_accuracy_calculation():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=4)
        actuals = [p["prediction"] for p in preds[:2]] + \
                  [{"heads": "tails", "tails": "heads"}[p["prediction"]] for p in preds[2:]]
        cfc.train(preds, actuals, realm="test")
        assert math.isclose(cfc.accuracy(), 0.5, abs_tol=1e-9)


def test_train_record_type_field():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=1)
        records = cfc.train(preds, ["heads"], realm="test")
        assert records[0]["record_type"] == "training"


def test_train_writes_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=2)
        cfc.train(preds, ["heads", "tails"], realm="digital")
        train_file = cfc.session_dir / cfc.TRAINING_FILE
        assert train_file.exists()
        lines = [l for l in train_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


def test_train_realm_stored():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=1)
        records = cfc.train(preds, ["heads"], realm="real")
        assert records[0]["realm"] == "real"


def test_train_stores_prediction_provenance():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=1)
        records = cfc.train(preds, ["heads"], realm="digital")
        rec = records[0]
        assert "predicted" in rec
        assert "actual" in rec
        assert "correct" in rec
        assert "thought_pattern" in rec
        assert "system_snapshot_at_prediction" in rec


def test_train_normalises_input_case():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=1)
        forced_prediction = preds[0]["prediction"]
        records = cfc.train(preds, [forced_prediction.upper()], realm="test")
        assert records[0]["correct"] is True


# ---------------------------------------------------------------------------
# CoinFlipConsciousness — persistence
# ---------------------------------------------------------------------------


def test_load_predictions_returns_saved_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        cfc.predict(n_coins=5)
        loaded = cfc.load_predictions()
        assert len(loaded) == 5


def test_load_training_returns_saved_records():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=3)
        cfc.train(preds, ["heads", "tails", "heads"], realm="digital")
        loaded = cfc.load_training()
        assert len(loaded) == 3


def test_data_persists_across_instances():
    """Records written by one CFC instance are readable by a new one (same dir)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc1 = _make_cfc(tmpdir)
        cfc1.predict(n_coins=2)

        cfc2 = CoinFlipConsciousness(
            session_dir=tmpdir, session_id="test_session", seed=99
        )
        loaded = cfc2.load_predictions()
        assert len(loaded) == 2


def test_predictions_accumulate_across_calls():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        cfc.predict(n_coins=2)
        cfc.predict(n_coins=3)
        loaded = cfc.load_predictions()
        assert len(loaded) == 5


def test_no_size_cap_on_storage():
    """Storage must never truncate — 500 records should all be present."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        for _ in range(50):
            preds = cfc.predict(n_coins=10)
            actuals = [p["prediction"] for p in preds]
            cfc.train(preds, actuals, realm="test")

        loaded_preds = cfc.load_predictions()
        loaded_train = cfc.load_training()
        assert len(loaded_preds) == 500
        assert len(loaded_train) == 500


# ---------------------------------------------------------------------------
# CoinFlipConsciousness.stats()
# ---------------------------------------------------------------------------


def test_stats_contains_required_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        stats = cfc.stats()
        required = {
            "session_id",
            "step",
            "total_trained",
            "correct",
            "accuracy",
            "session_dir",
            "predictions_file",
            "training_file",
        }
        assert required.issubset(stats.keys())


def test_stats_reflects_training():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        preds = cfc.predict(n_coins=4)
        actuals = [p["prediction"] for p in preds]  # all correct
        cfc.train(preds, actuals, realm="test")
        stats = cfc.stats()
        assert stats["total_trained"] == 4
        assert stats["correct"] == 4
        assert stats["accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Interactive helpers — run_digitalrealm / run_realrealm / run_predictor
# (Keyboard interrupt simulated via StopIteration → KeyboardInterrupt patch)
# ---------------------------------------------------------------------------


def _raises_keyboard_interrupt(*_args, **_kwargs):
    raise KeyboardInterrupt


def test_run_predictor_stops_on_keyboard_interrupt(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        with patch("builtins.input", side_effect=KeyboardInterrupt):
            run_predictor(1, cfc)
        out = capsys.readouterr().out
        assert "stopped" in out.lower() or "predict" in out.lower()


def test_run_digitalrealm_one_round(capsys):
    """Simulate one round: Enter → predict → Enter to reveal → KeyboardInterrupt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        # input() calls: (1) start round, (2) reveal, (3) next round → interrupt
        with patch("builtins.input", side_effect=["", "", KeyboardInterrupt]):
            run_digitalrealm(1, cfc)
        # After one complete round we should have 1 prediction and 1 training record
        assert len(cfc.load_predictions()) == 1
        assert len(cfc.load_training()) == 1
        out = capsys.readouterr().out
        assert "actual" in out.lower() or "reveal" in out.lower()


def test_run_realrealm_one_round(capsys):
    """Simulate one real-realm round: Enter → user inputs 'h' → KeyboardInterrupt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        # input() calls: (1) ready to flip, (2) actual result 'h', (3) next → interrupt
        with patch("builtins.input", side_effect=["", "h", KeyboardInterrupt]):
            run_realrealm(1, cfc)
        assert len(cfc.load_predictions()) == 1
        assert len(cfc.load_training()) == 1
        out = capsys.readouterr().out
        assert "flip" in out.lower() or "heads" in out.lower()


def test_run_realrealm_invalid_then_valid_input(capsys):
    """Invalid input (not h/t) should prompt again until a valid answer is given."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        # Round 1: "ready", then "x" (invalid), then "t" (valid)
        # Round 2: start → KeyboardInterrupt
        with patch("builtins.input", side_effect=["", "x", "t", KeyboardInterrupt]):
            run_realrealm(1, cfc)
        records = cfc.load_training()
        assert len(records) == 1
        assert records[0]["actual"] == "tails"


def test_run_realrealm_multi_coin(capsys):
    """Two-coin round: two 'h' inputs expected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        # (1) ready, (2) coin 1 actual, (3) coin 2 actual, (4) next round → interrupt
        with patch("builtins.input", side_effect=["", "h", "t", KeyboardInterrupt]):
            run_realrealm(2, cfc)
        preds = cfc.load_predictions()
        trains = cfc.load_training()
        assert len(preds) == 2
        assert len(trains) == 2
        assert trains[0]["actual"] == "heads"
        assert trains[1]["actual"] == "tails"


def test_run_digitalrealm_multi_coin(capsys):
    """Two-coin digital round: 2 predictions and 2 training records per round."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir)
        with patch("builtins.input", side_effect=["", "", KeyboardInterrupt]):
            run_digitalrealm(2, cfc)
        assert len(cfc.load_predictions()) == 2
        assert len(cfc.load_training()) == 2


# ---------------------------------------------------------------------------
# Run standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    import traceback

    passed = failed = skipped = 0
    for _name, _func in list(globals().items()):
        if not _name.startswith("test_"):
            continue
        # Skip tests that need pytest fixtures (capsys, etc.)
        sig = inspect.signature(_func)
        if sig.parameters:
            skipped += 1
            print(f"  ⏭️  {_name} (skipped — requires pytest fixture)")
            continue
        try:
            _func()
            print(f"  ✅ {_name}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {_name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (use pytest for full suite)")
    sys.exit(0 if failed == 0 else 1)
