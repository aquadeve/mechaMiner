#!/usr/bin/env python3
"""
Tests for the --remoteview / .binai / sensory-bridge pipeline.

Coverage
--------
* BinaiWriter / BinaiReader round-trips (pure Python, no hardware).
* AudioToLetters.convert — strips non-letters, normalises case.
* letters_to_vector — correct length, values in [-1, 1].
* geometry_to_vector — correct length, values clamped.
* sensory_fusion_to_vector — 80-dim output, all channels represented.
* EmulatedEars in non-interactive mode — returns AudioObservation with source="none".
* EmulatedVision — returns VisionFrame with synthetic geometry.
* CoinFlipConsciousness with enable_sensory=False — unchanged behaviour.
* CoinFlipConsciousness with enable_sensory=True (mocked sensors) — sensory
  data appears in prediction records and consciousness is updated.
* export_binai() — file written, readable, correct content.
* Accuracy threshold auto-export — .binai created exactly once when threshold met.
* _print_prediction shows audio/vision lines when present (capsys).
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Path setup (mirrors conftest.py)
# ---------------------------------------------------------------------------
_here = os.path.dirname(__file__)
_src = os.path.join(_here, "..", "src")
for _d in [_src, os.path.join(_src, "fsot_core"), os.path.join(_src, "neuromorphic_brain")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from binai_writer import MAGIC, BinaiReader, BinaiWriter
from sensory_bridge import (
    AudioObservation,
    AudioToLetters,
    EmulatedEars,
    EmulatedVision,
    VisionFrame,
    geometry_to_vector,
    letters_to_vector,
    sensory_fusion_to_vector,
)
from coin_flip_consciousness import (
    CoinFlipConsciousness,
    _print_prediction,
)
from core_system import ConsciousnessState


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_cfc(
    tmpdir: str,
    enable_sensory: bool = False,
    accuracy_threshold: float = 0.0,
) -> CoinFlipConsciousness:
    return CoinFlipConsciousness(
        session_dir=tmpdir,
        session_id="test_rv",
        seed=7,
        enable_sensory=enable_sensory,
        accuracy_threshold=accuracy_threshold,
    )


def _fake_audio(letters: str = "heads") -> AudioObservation:
    return AudioObservation(
        raw_text=letters,
        letters=AudioToLetters.convert(letters),
        source="typed",
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def _fake_vision(brightness: float = 128.0) -> VisionFrame:
    return VisionFrame(
        jpeg_bytes=b"",
        geometry={"mean_brightness": brightness, "phase": 0.5, "node_hash_hex": "aabbccdd"},
        source="synthetic",
        captured_at_utc="2026-01-01T00:00:00+00:00",
    )


def _minimal_snapshot() -> Dict[str, Any]:
    import time
    return {
        "unix_timestamp": time.time(),
        "local_microsecond": 500000,
        "local_minute": 30,
        "local_hour": 12,
        "local_second": 45,
        "platform_node": "testhost",
        "utc_iso": "2026-01-01T00:00:00+00:00",
        "monotonic_ns": 123456789,
        "local_year": 2026, "local_month": 1, "local_day": 1,
        "platform_system": "Linux", "platform_machine": "x86_64",
    }


# ===========================================================================
# AudioToLetters
# ===========================================================================

def test_audio_to_letters_lowercase():
    assert AudioToLetters.convert("HEADS") == "heads"


def test_audio_to_letters_strips_digits_and_punctuation():
    result = AudioToLetters.convert("Hello! 123 World.")
    assert result == "helloworld"


def test_audio_to_letters_empty_string():
    assert AudioToLetters.convert("") == ""


def test_audio_to_letters_numbers_only():
    assert AudioToLetters.convert("42 99 !!") == ""


def test_audio_to_letters_all_letters():
    assert AudioToLetters.convert("abcxyz") == "abcxyz"


def test_audio_to_letters_unicode_stripped():
    # Non-ASCII chars must be stripped
    result = AudioToLetters.convert("café")
    assert all(c in "abcdefghijklmnopqrstuvwxyz" for c in result)


# ===========================================================================
# letters_to_vector
# ===========================================================================

def test_letters_to_vector_correct_dim():
    assert len(letters_to_vector("hello", dim=20)) == 20


def test_letters_to_vector_custom_dim():
    assert len(letters_to_vector("hello", dim=7)) == 7


def test_letters_to_vector_range():
    vec = letters_to_vector("abcdefghijklmnopqrstuvwxyz", dim=26)
    assert all(-1.0 <= v <= 1.0 for v in vec)


def test_letters_to_vector_empty_returns_zeros():
    vec = letters_to_vector("", dim=10)
    assert vec == [0.0] * 10


def test_letters_to_vector_single_letter_has_max():
    # Only 'a' present → that bucket should produce the max (1.0)
    vec = letters_to_vector("aaaa", dim=26)
    assert max(vec) == pytest_approx(1.0)


# ===========================================================================
# geometry_to_vector
# ===========================================================================

def test_geometry_to_vector_correct_dim():
    assert len(geometry_to_vector({}, dim=20)) == 20


def test_geometry_to_vector_custom_dim():
    assert len(geometry_to_vector({"mean_brightness": 200.0}, dim=5)) == 5


def test_geometry_to_vector_range():
    geo = {"mean_brightness": 255.0, "mean_r": 0.0, "mean_g": 128.0,
           "mean_b": 200.0, "width": 640, "height": 480, "phase": 0.5}
    vec = geometry_to_vector(geo, dim=20)
    assert all(-1.1 <= v <= 1.1 for v in vec), f"out of range: {vec}"


def test_geometry_to_vector_empty_geo():
    vec = geometry_to_vector({}, dim=10)
    assert len(vec) == 10


def test_geometry_to_vector_node_hash_decoded():
    geo = {"node_hash_hex": "ff00aa55"}
    vec = geometry_to_vector(geo, dim=20)
    # Byte 0xff → 1.0, byte 0x00 → -1.0
    assert vec[7] == pytest_approx(1.0, abs=0.01)
    assert vec[8] == pytest_approx(-1.0, abs=0.01)


# ===========================================================================
# sensory_fusion_to_vector
# ===========================================================================

def test_sensory_fusion_correct_dim():
    snap = _minimal_snapshot()
    vec = sensory_fusion_to_vector(snap, None, None, total_dim=80)
    assert len(vec) == ConsciousnessState.TOTAL_DIM


def test_sensory_fusion_range():
    snap = _minimal_snapshot()
    audio = _fake_audio("headsortails")
    vision = _fake_vision(200.0)
    vec = sensory_fusion_to_vector(snap, audio, vision, total_dim=80)
    assert all(-1.1 <= v <= 1.1 for v in vec), f"out of range"


def test_sensory_fusion_without_sensory_still_80_dims():
    snap = _minimal_snapshot()
    vec = sensory_fusion_to_vector(snap, None, None, total_dim=80)
    assert len(vec) == 80


def test_sensory_fusion_audio_changes_vector():
    snap = _minimal_snapshot()
    vec_none = sensory_fusion_to_vector(snap, None, None)
    vec_audio = sensory_fusion_to_vector(snap, _fake_audio("aaaa"), None)
    # Audio band (dims 20-39) should differ
    assert vec_none[20:40] != vec_audio[20:40]


def test_sensory_fusion_vision_changes_vector():
    snap = _minimal_snapshot()
    vec_none = sensory_fusion_to_vector(snap, None, None)
    vec_vis = sensory_fusion_to_vector(snap, None, _fake_vision(255.0))
    # Vision band (dims 40-59) should differ
    assert vec_none[40:60] != vec_vis[40:60]


def test_sensory_fusion_cross_modal_is_product():
    snap = _minimal_snapshot()
    audio = _fake_audio("aaaa")
    vision = _fake_vision(128.0)
    vec = sensory_fusion_to_vector(snap, audio, vision, total_dim=80)
    audio_band = vec[20:40]
    vision_band = vec[40:60]
    cross_band = vec[60:80]
    for i in range(20):
        expected = audio_band[i] * vision_band[i]
        assert math.isclose(cross_band[i], expected, abs_tol=1e-9), (
            f"cross_band[{i}] = {cross_band[i]}, expected {expected}"
        )


# ===========================================================================
# EmulatedEars (non-interactive, no hardware)
# ===========================================================================

def test_emulated_ears_non_interactive_source_none():
    ears = EmulatedEars(interactive=False, preferred_source="none")
    obs = ears.capture()
    assert obs.source == "none"
    assert obs.letters == ""


def test_emulated_ears_observation_has_timestamp():
    ears = EmulatedEars(interactive=False)
    obs = ears.capture()
    assert obs.captured_at_utc != ""


def test_emulated_ears_to_dict_keys():
    ears = EmulatedEars(interactive=False)
    obs = ears.capture()
    d = obs.to_dict()
    assert {"raw_text", "letters", "source", "captured_at_utc"}.issubset(d.keys())


def test_emulated_ears_typed_input(capsys):
    ears = EmulatedEars(interactive=True, preferred_source="none")
    with patch("builtins.input", return_value="HEADS tails"):
        obs = ears.capture()
    assert obs.source == "typed"
    assert obs.letters == "headstails"
    assert obs.raw_text == "HEADS tails"


def test_emulated_ears_eof_returns_none_source():
    ears = EmulatedEars(interactive=True, preferred_source="none")
    with patch("builtins.input", side_effect=EOFError):
        obs = ears.capture()
    assert obs.source == "none"


# ===========================================================================
# EmulatedVision (no hardware — synthetic fallback)
# ===========================================================================

def test_emulated_vision_returns_vision_frame():
    vision = EmulatedVision()
    frame = vision.capture()
    assert isinstance(frame, VisionFrame)


def test_emulated_vision_synthetic_geometry_keys():
    vision = EmulatedVision()
    frame = vision.capture()
    geo = frame.geometry
    assert "mean_brightness" in geo


def test_emulated_vision_source_is_synthetic_or_camera():
    vision = EmulatedVision()
    frame = vision.capture()
    assert frame.source in ("synthetic", "camera")


def test_emulated_vision_has_timestamp():
    vision = EmulatedVision()
    frame = vision.capture()
    assert frame.captured_at_utc != ""


def test_emulated_vision_to_dict_keys():
    frame = _fake_vision()
    d = frame.to_dict()
    assert {"jpeg_bytes_size", "geometry", "source", "captured_at_utc"}.issubset(d.keys())


# ===========================================================================
# BinaiWriter / BinaiReader
# ===========================================================================

def test_binai_magic_bytes():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.binai"
        BinaiWriter().write(
            path=path,
            session_id="s1",
            accuracy=0.75,
            threshold=0.65,
            total_trained=10,
            correct=7,
            consciousness_vector=[0.5] * 80,
        )
        raw = path.read_bytes()
        assert raw[:6] == MAGIC


def test_binai_round_trip_scalars():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "snap.binai"
        BinaiWriter().write(
            path=path,
            session_id="sess_42",
            accuracy=0.8,
            threshold=0.7,
            total_trained=50,
            correct=40,
            consciousness_vector=[float(i) / 80.0 for i in range(80)],
            audio_letters="headsortails",
            audio_source="typed",
        )
        payload = BinaiReader().read(path)
        assert payload["session_id"] == "sess_42"
        assert math.isclose(payload["accuracy_at_export"], 0.8, abs_tol=1e-6)
        assert math.isclose(payload["threshold"], 0.7, abs_tol=1e-6)
        assert payload["total_trained"] == 50
        assert payload["correct"] == 40
        assert payload["audio_letters"] == "headsortails"
        assert payload["audio_source"] == "typed"


def test_binai_round_trip_consciousness_vector():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "vec.binai"
        orig_vec = [float(i) * 0.01 - 0.4 for i in range(80)]
        BinaiWriter().write(
            path=path,
            session_id="v",
            accuracy=0.5,
            threshold=0.5,
            total_trained=2,
            correct=1,
            consciousness_vector=orig_vec,
        )
        payload = BinaiReader().read(path)
        for a, b in zip(orig_vec, payload["consciousness_vector"]):
            assert math.isclose(a, b, abs_tol=1e-12)


def test_binai_round_trip_vision_jpeg():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "img.binai"
        fake_jpeg = b"\xff\xd8\xff" + b"\x00" * 100  # fake JPEG header
        BinaiWriter().write(
            path=path,
            session_id="j",
            accuracy=0.9,
            threshold=0.8,
            total_trained=20,
            correct=18,
            consciousness_vector=[0.0] * 80,
            vision_jpeg=fake_jpeg,
        )
        payload = BinaiReader().read(path)
        assert payload["vision_jpeg"] == fake_jpeg
        assert payload["has_vision_jpeg"] is True


def test_binai_empty_vision_jpeg():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "novid.binai"
        BinaiWriter().write(
            path=path,
            session_id="nv",
            accuracy=0.6,
            threshold=0.5,
            total_trained=5,
            correct=3,
            consciousness_vector=[0.0] * 80,
        )
        payload = BinaiReader().read(path)
        assert payload["vision_jpeg"] == b""
        assert payload["has_vision_jpeg"] is False


def test_binai_extra_metadata_round_trips():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "meta.binai"
        BinaiWriter().write(
            path=path,
            session_id="m",
            accuracy=0.5,
            threshold=0.4,
            total_trained=1,
            correct=0,
            consciousness_vector=[0.1] * 80,
            extra_metadata={"custom_key": "custom_value", "step": 99},
        )
        payload = BinaiReader().read(path)
        assert payload["custom_key"] == "custom_value"
        assert payload["step"] == 99


def test_binai_reader_rejects_bad_magic():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bad.binai"
        path.write_bytes(b"NOTBINAI" + b"\x00" * 20)
        try:
            BinaiReader().read(path)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "magic" in str(e).lower() or "invalid" in str(e).lower()


def test_binai_reader_rejects_too_short():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "tiny.binai"
        path.write_bytes(b"\x00\x01")
        try:
            BinaiReader().read(path)
            assert False, "Expected ValueError"
        except ValueError:
            pass


# ===========================================================================
# CoinFlipConsciousness — sensory disabled (existing behaviour unchanged)
# ===========================================================================

def test_cfc_sensory_disabled_prediction_keys():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir, enable_sensory=False)
        pred = cfc.predict(1)[0]
        assert "audio_observation" in pred
        assert "vision_observation" in pred
        assert pred["sensory_enabled"] is False


def test_cfc_sensory_disabled_audio_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir, enable_sensory=False)
        pred = cfc.predict(1)[0]
        assert pred["audio_observation"] == {}


# ===========================================================================
# CoinFlipConsciousness — sensory enabled (mocked hardware)
# ===========================================================================

def _make_cfc_with_mocked_sensors(tmpdir: str) -> CoinFlipConsciousness:
    """Build a CFC with enable_sensory=True but mocked ears and vision."""
    cfc = CoinFlipConsciousness(
        session_dir=tmpdir,
        session_id="test_sensory",
        seed=11,
        enable_sensory=True,
        interactive_sensory=False,
    )
    # Replace the real organs with mocks that return deterministic data
    cfc._ears = MagicMock()
    cfc._ears.capture.return_value = _fake_audio("tails")
    cfc._vision = MagicMock()
    cfc._vision.capture.return_value = _fake_vision(100.0)
    return cfc


def test_cfc_sensory_enabled_prediction_has_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        pred = cfc.predict(1)[0]
        assert pred["sensory_enabled"] is True
        audio = pred["audio_observation"]
        assert audio.get("letters") == "tails"
        assert audio.get("source") == "typed"


def test_cfc_sensory_enabled_prediction_has_vision():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        pred = cfc.predict(1)[0]
        vision = pred["vision_observation"]
        geo = vision.get("geometry", {})
        assert math.isclose(geo.get("mean_brightness", 0), 100.0, abs_tol=0.01)


def test_cfc_sensory_enabled_organs_called_each_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        cfc.predict(3)
        assert cfc._ears.capture.call_count == 3
        assert cfc._vision.capture.call_count == 3


def test_cfc_sensory_fusion_changes_consciousness():
    """With mocked sensory inputs the consciousness vector should differ from
    a no-sensor run even with the same seed and time step."""
    with tempfile.TemporaryDirectory() as td1, tempfile.TemporaryDirectory() as td2:
        cfc_plain = CoinFlipConsciousness(
            session_dir=td1, session_id="plain", seed=11, enable_sensory=False
        )
        cfc_sensory = _make_cfc_with_mocked_sensors(td2)

        plain_pred = cfc_plain.predict(1)[0]
        sens_pred = cfc_sensory.predict(1)[0]

        # Sensory-enriched vector should differ in at least the audio/vision bands
        plain_vec = plain_pred["consciousness_vector"]
        sens_vec = sens_pred["consciousness_vector"]
        # Dims 20-79 include audio, vision, and cross-modal bands
        assert plain_vec[20:] != sens_vec[20:], (
            "Sensory input should change consciousness vector dims 20-79"
        )


# ===========================================================================
# export_binai
# ===========================================================================

def test_export_binai_creates_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        cfc.predict(1)
        path = cfc.export_binai()
        assert Path(path).exists()
        assert Path(path).suffix == ".binai"


def test_export_binai_readable():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        cfc.predict(2)
        preds = cfc.load_predictions()
        actuals = [p["prediction"] for p in preds]
        cfc.train(preds, actuals, realm="test")
        path = cfc.export_binai()
        payload = BinaiReader().read(path)
        assert payload["session_id"] == "test_sensory"
        assert len(payload["consciousness_vector"]) == ConsciousnessState.TOTAL_DIM


def test_export_binai_accuracy_reflects_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        preds = cfc.predict(4)
        # Force 3/4 correct
        actuals = [p["prediction"] for p in preds[:3]] + [
            {"heads": "tails", "tails": "heads"}[preds[3]["prediction"]]
        ]
        cfc.train(preds, actuals, realm="test")
        path = cfc.export_binai()
        payload = BinaiReader().read(path)
        assert math.isclose(payload["accuracy_at_export"], 0.75, abs_tol=1e-6)


def test_export_binai_stores_audio_letters():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        cfc.predict(1)  # captures fake audio "tails" → "tails"
        path = cfc.export_binai()
        payload = BinaiReader().read(path)
        assert payload["audio_letters"] == "tails"


def test_export_binai_stores_vision_geometry():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc_with_mocked_sensors(tmpdir)
        cfc.predict(1)
        path = cfc.export_binai()
        payload = BinaiReader().read(path)
        assert "mean_brightness" in payload["vision_geometry"]


def test_export_binai_no_sensory_still_works():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir, enable_sensory=False)
        cfc.predict(1)
        path = cfc.export_binai()
        payload = BinaiReader().read(path)
        assert payload["audio_letters"] == ""
        assert payload["audio_source"] == "none"


def test_export_binai_custom_out_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir, enable_sensory=False)
        cfc.predict(1)
        custom = os.path.join(tmpdir, "exports")
        path = cfc.export_binai(out_dir=custom)
        assert str(path).startswith(custom)
        assert Path(path).exists()


# ===========================================================================
# Accuracy threshold auto-export
# ===========================================================================

def test_threshold_auto_export_fires_at_threshold():
    """When accuracy reaches the threshold a .binai file must be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = CoinFlipConsciousness(
            session_dir=tmpdir,
            session_id="thresh_test",
            seed=5,
            enable_sensory=False,
            accuracy_threshold=0.5,
        )
        preds = cfc.predict(4)
        # Force 3/4 correct so accuracy = 0.75 ≥ 0.5
        actuals = [p["prediction"] for p in preds[:3]] + [
            {"heads": "tails", "tails": "heads"}[preds[3]["prediction"]]
        ]
        cfc.train(preds, actuals, realm="test")

        binai_files = list(Path(cfc.session_dir).glob("*.binai"))
        assert len(binai_files) == 1, f"Expected 1 .binai file, got {binai_files}"


def test_threshold_auto_export_only_once():
    """Even after further training the .binai is only exported once."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = CoinFlipConsciousness(
            session_dir=tmpdir,
            session_id="once_test",
            seed=5,
            enable_sensory=False,
            accuracy_threshold=0.5,
        )
        for _ in range(3):
            preds = cfc.predict(4)
            actuals = [p["prediction"] for p in preds]  # all correct
            cfc.train(preds, actuals, realm="test")

        binai_files = list(Path(cfc.session_dir).glob("*.binai"))
        assert len(binai_files) == 1


def test_threshold_zero_no_auto_export():
    """Threshold = 0 means disabled — no .binai should be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = _make_cfc(tmpdir, accuracy_threshold=0.0)
        preds = cfc.predict(4)
        actuals = [p["prediction"] for p in preds]
        cfc.train(preds, actuals, realm="test")

        binai_files = list(Path(cfc.session_dir).glob("*.binai"))
        assert len(binai_files) == 0


def test_threshold_not_met_no_export():
    """If accuracy stays below threshold no .binai should be created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfc = CoinFlipConsciousness(
            session_dir=tmpdir,
            session_id="below_thresh",
            seed=5,
            enable_sensory=False,
            accuracy_threshold=0.99,  # unreachable in 4 flips
        )
        preds = cfc.predict(4)
        # Force 0/4 correct
        actuals = [
            {"heads": "tails", "tails": "heads"}[p["prediction"]] for p in preds
        ]
        cfc.train(preds, actuals, realm="test")

        binai_files = list(Path(cfc.session_dir).glob("*.binai"))
        assert len(binai_files) == 0


# ===========================================================================
# _print_prediction — sensory display
# ===========================================================================

def test_print_prediction_shows_audio(capsys):
    pred = {
        "system_snapshot": {"utc_iso": "2026-01-01T00:00:00+00:00"},
        "coin_index": 1,
        "n_coins": 1,
        "prediction": "heads",
        "thought_confidence": 0.65,
        "audio_observation": {"letters": "headsortails", "source": "typed"},
        "vision_observation": {},
    }
    _print_prediction(pred)
    out = capsys.readouterr().out
    assert "headsortails" in out
    assert "typed" in out


def test_print_prediction_shows_vision(capsys):
    pred = {
        "system_snapshot": {"utc_iso": "2026-01-01T00:00:00+00:00"},
        "coin_index": 1,
        "n_coins": 1,
        "prediction": "tails",
        "thought_confidence": 0.55,
        "audio_observation": {},
        "vision_observation": {
            "geometry": {"mean_brightness": 127.5},
            "source": "synthetic",
        },
    }
    _print_prediction(pred)
    out = capsys.readouterr().out
    assert "127.5" in out or "brightness" in out.lower()


def test_print_prediction_no_sensory_no_extra_lines(capsys):
    pred = {
        "system_snapshot": {"utc_iso": "2026-01-01T00:00:00+00:00"},
        "coin_index": 1,
        "n_coins": 1,
        "prediction": "heads",
        "thought_confidence": 0.7,
        "audio_observation": {},
        "vision_observation": {},
    }
    _print_prediction(pred)
    out = capsys.readouterr().out
    # No extra sensory lines
    assert "👂" not in out
    assert "👁" not in out


# ===========================================================================
# pytest_approx compat shim
# ===========================================================================

def pytest_approx(expected, abs=None, rel=None):
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


# ===========================================================================
# Standalone runner (skips fixture-dependent tests)
# ===========================================================================

if __name__ == "__main__":
    import inspect
    import traceback

    passed = failed = skipped = 0
    for _name, _func in list(globals().items()):
        if not _name.startswith("test_"):
            continue
        sig = inspect.signature(_func)
        if sig.parameters:
            skipped += 1
            print(f"  ⏭️  {_name} (requires pytest fixture)")
            continue
        try:
            _func()
            print(f"  ✅ {_name}")
            passed += 1
        except Exception as exc:
            print(f"  ❌ {_name}: {exc}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(0 if failed == 0 else 1)
