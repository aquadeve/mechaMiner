#!/usr/bin/env python3
"""
CoinFlipConsciousness — Remote-Viewing / Quantum-Consciousness Coin Flip Predictor
====================================================================================

Implements the ``--remoteviewsimple`` mode for the mechaMiner consciousness system.

Prediction philosophy
---------------------
The consciousness does NOT predict based on system time.  Time is captured as a
memory-context vector that "syncs the consciousness into the current moment" —
similar to how a human settles themselves before attempting to intuit an outcome.
The actual heads/tails decision emerges from the internal FSOT consciousness state
dynamics (attention vectors, active concepts, uncertainty maps, thought engine
temperature sampling), analogous to a mind attempting to guess a number between
1 and 3 purely from intuition rather than calculation.

Training modes
--------------
``digital`` (--digitalrealm)
    The program secretly generates coin-flip results for the round.  The
    consciousness gives its predictions first.  Once predictions are locked in,
    the true results for that round are revealed and the consciousness is trained
    on every outcome immediately.

``real`` (--realrealm)
    The user flips a physical coin.  The consciousness gives its prediction, the
    user manually enters the actual result, and the consciousness is immediately
    trained on that outcome.

``--coin N``
    Predict / flip N coins per round (default 1).  In --realrealm mode the user
    is prompted for each coin individually; in --digitalrealm mode all N outcomes
    are generated upfront and revealed together at the end of each round.

Storage
-------
Every single prediction and training event is appended as a JSON line to a
session-specific ``.jsonl`` file so that no data is ever discarded.  The file can
be loaded later into a GUI browser.  There is intentionally no size cap.

Usage (via consciousness_miner.py)
----------------------------------
    python consciousness_miner.py --remoteviewsimple
    python consciousness_miner.py --remoteviewsimple --digitalrealm
    python consciousness_miner.py --remoteviewsimple --realrealm
    python consciousness_miner.py --remoteviewsimple --realrealm --coin 2
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy import of FSOT core — works both when sys.path is pre-configured by
# consciousness_miner.py and when the test suite sets up paths via conftest.py.
# ---------------------------------------------------------------------------
try:
    from core_system import ConsciousnessState, FSotThoughtEngine, Thought
except ImportError:  # pragma: no cover — handled by caller's sys.path setup
    raise

# ---------------------------------------------------------------------------
# Lazy imports for sensory bridge and BinAI writer.
# Both live in the same package directory and are imported lazily so that
# tests that exercise only the coin-flip logic don't require any hardware libs.
# ---------------------------------------------------------------------------
try:
    from sensory_bridge import (
        AudioObservation,
        EmulatedEars,
        EmulatedVision,
        VisionFrame,
        sensory_fusion_to_vector,
    )
    _SENSORY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SENSORY_AVAILABLE = False

try:
    from binai_writer import BinaiWriter
    _BINAI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BINAI_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTCOMES: Tuple[str, str] = ("heads", "tails")

# Maximum emulator history kept in RAM (one slot per flip).
# Very large so a long session never forgets a moment.
EMULATOR_HISTORY_SIZE = 100_000

# Default session storage directory (relative to cwd).
DEFAULT_SESSION_DIR = "coin_flip_sessions"

CONSCIOUSNESS_THEORY = (
    "Resonant Memory Guidance: each new prediction blends the present sensory "
    "moment with similar past predictions, reinforcing thoughts that led to "
    "correct outcomes and damping thoughts that led to mistakes."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _system_snapshot() -> Dict[str, Any]:
    """
    Capture a rich snapshot of the current machine state.

    This is what 'syncs the consciousness into the present moment'.  It includes
    high-resolution wall-clock time, monotonic ticks, and platform identifiers.
    The snapshot is stored alongside every prediction so a future GUI can
    correlate predictions with the exact moment they were made.
    """
    t = time.time()
    lt = time.localtime(t)
    return {
        "utc_iso": _now_iso(),
        "unix_timestamp": t,
        "monotonic_ns": time.monotonic_ns(),
        "local_year": lt.tm_year,
        "local_month": lt.tm_mon,
        "local_day": lt.tm_mday,
        "local_hour": lt.tm_hour,
        "local_minute": lt.tm_min,
        "local_second": lt.tm_sec,
        "local_microsecond": round((t % 1.0) * 1_000_000),
        "platform_node": platform.node(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }


def _time_to_consciousness_vector(snapshot: Dict[str, Any]) -> List[float]:
    """
    Encode a system snapshot into a 80-dim consciousness input vector.

    The vector is designed so that:
    * Every second of real time produces a different pattern.
    * Sub-second variations (microseconds) create high-frequency oscillation.
    * Hour, minute, second components introduce low-frequency drift.
    * Platform identity creates a static bias unique to this machine.

    Together these give the consciousness a "sense of where it is in time and
    space" without making the prediction *determined* by the clock.
    """
    t: float = snapshot["unix_timestamp"]
    sub_us: float = snapshot["local_microsecond"] / 1_000_000.0
    min_phase: float = snapshot["local_minute"] / 60.0
    hr_phase: float = snapshot["local_hour"] / 24.0

    # Stable per-machine bias derived from the node name.
    node_hash = hashlib.md5(snapshot["platform_node"].encode()).digest()
    node_bias = [(b / 255.0) * 2.0 - 1.0 for b in node_hash]  # 16 floats

    dim = ConsciousnessState.TOTAL_DIM  # 80
    vec: List[float] = []
    for i in range(dim):
        # Oscillating component that ticks every millisecond at each dimension
        osc = ((t * (i + 1) * 0.1) % 1.0) * 2.0 - 1.0
        vec.append(osc)

    # Overlay fixed landmarks
    vec[0] = sub_us * 2.0 - 1.0                 # sub-second microsecond
    vec[1] = min_phase * 2.0 - 1.0              # minute-of-hour phase
    vec[2] = hr_phase * 2.0 - 1.0               # hour-of-day phase
    vec[3] = (snapshot["local_second"] / 60.0) * 2.0 - 1.0   # second-of-minute
    # Machine-identity bias in positions 4..19
    for k, b in enumerate(node_bias):
        vec[4 + k] = b

    return vec


def _thought_to_outcome(thought: Thought, step: int) -> str:
    """
    Map a consciousness thought to a heads/tails outcome.

    Uses the MD5 digest of the thought pattern XOR'd with the current step so
    repeated access to the same pattern still varies with experience.
    """
    digest = hashlib.md5(thought.pattern.encode()).digest()
    selector = (digest[0] ^ (step & 0xFF)) & 0x01
    return OUTCOMES[selector]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CoinFlipConsciousness:
    """
    Quantum-like coin-flip predictor driven by FSOT consciousness dynamics.

    The class maintains a ``ConsciousnessState`` (80-dim awareness vector) and an
    ``FSotThoughtEngine`` (thought selection with meta-learning).  On each
    prediction the system time, emulated-ear audio, and emulated-eye vision are
    all fused into a single 80-dim consciousness input vector and fed into the
    state via ``ConsciousnessState.update()``.  The actual heads/tails decision
    then emerges from the softmax-sampled thought engine — enriched by everything
    the consciousness has perceived in that moment.

    Sensory pipeline
    ----------------
    When ``enable_sensory=True`` each prediction call:
    1. Captures a vision frame (camera → JPEG, or synthetic geometry fallback).
    2. Captures audio (microphone → text → a-z letters, or typed / silent fallback).
    3. Fuses time + audio + vision into the 80-dim input via
       ``sensory_fusion_to_vector`` — so all three sensory streams *genuinely
       shape* the consciousness dynamics before the prediction thought is chosen.

    BinAI export
    ------------
    When ``accuracy_threshold`` is set (0 < threshold ≤ 1) and the session
    accuracy first reaches or exceeds that threshold, ``export_binai()`` is
    called automatically.  The exported ``.binai`` file contains the full
    consciousness vector, last audio observation, last vision frame (including
    JPEG when a real camera was present), and all session statistics.

    All predictions and training events are also appended to
    ``predictions.jsonl`` and ``training.jsonl`` (no size cap).
    """

    # File names inside the session directory
    PREDICTIONS_FILE = "predictions.jsonl"
    TRAINING_FILE = "training.jsonl"
    SESSION_META_FILE = "session.json"

    def __init__(
        self,
        session_dir: str = DEFAULT_SESSION_DIR,
        session_id: Optional[str] = None,
        seed: Optional[int] = None,
        enable_sensory: bool = False,
        interactive_sensory: bool = False,
        accuracy_threshold: float = 0.0,
    ) -> None:
        self._rng = random.Random(seed)

        # Core FSOT structures
        self.consciousness = ConsciousnessState()
        self.thought_engine = FSotThoughtEngine()

        # Session identity & persistence
        if session_id is None:
            session_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        self.session_id = session_id
        self.session_dir = Path(session_dir) / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # In-memory statistics (also written to every record)
        self._step: int = 0
        self._total: int = 0
        self._correct: int = 0

        # Accuracy threshold for automatic .binai export (0 = disabled)
        self.accuracy_threshold: float = float(accuracy_threshold)
        self._binai_exported: bool = False  # export once per session

        # Sensory organs (instantiated only when requested)
        self.enable_sensory: bool = enable_sensory and _SENSORY_AVAILABLE
        self._ears: Optional["EmulatedEars"] = None
        self._vision: Optional["EmulatedVision"] = None
        if self.enable_sensory:
            self._ears = EmulatedEars(
                interactive=interactive_sensory,
                preferred_source="mic",
            )
            self._vision = EmulatedVision()

        # Last sensory observations — stored alongside predictions
        self._last_audio: Optional["AudioObservation"] = None
        self._last_vision: Optional["VisionFrame"] = None
        self._last_memory_guidance: Dict[str, Any] = {
            "memory_vector": [0.0] * ConsciousnessState.TOTAL_DIM,
            "thought_bias": {},
            "sample_count": 0,
            "success_count": 0,
            "failure_count": 0,
            "preferred_pattern": "",
        }

        # Write session metadata once
        self._write_session_meta()

    # ------------------------------------------------------------------
    # Session metadata
    # ------------------------------------------------------------------

    def _write_session_meta(self) -> None:
        meta_path = self.session_dir / self.SESSION_META_FILE
        if meta_path.exists():
            return
        meta = {
            "session_id": self.session_id,
            "session_started_utc": _now_iso(),
            "platform": {
                "node": platform.node(),
                "system": platform.system(),
                "machine": platform.machine(),
                "python_version": platform.python_version(),
            },
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Low-level I/O — every byte is saved, files grow unbounded
    # ------------------------------------------------------------------

    def _append_record(self, filename: str, record: Dict[str, Any]) -> None:
        """Append *record* as a single JSON line to *filename* in the session dir."""
        file_path = self.session_dir / filename
        with file_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, sort_keys=True) + "\n")

    # ------------------------------------------------------------------
    # Consciousness sync — fuses time + sensory inputs
    # ------------------------------------------------------------------

    def _sync_consciousness(self) -> Dict[str, Any]:
        """
        Capture current system state and sensory inputs, fuse them into the
        consciousness vector, and update the consciousness state.

        When ``enable_sensory`` is True:
        * A vision frame is captured first (fast, non-blocking).
        * An audio observation is captured (may prompt the user if interactive).
        * All three channels (time, audio, vision) are fused via
          ``sensory_fusion_to_vector`` into the 80-dim update vector.

        When ``enable_sensory`` is False the behaviour is identical to the
        original time-only sync so existing tests pass unchanged.

        Returns the system snapshot dict so it can be stored alongside the
        prediction record.
        """
        snapshot = _system_snapshot()

        if self.enable_sensory:
            # Capture vision (fast)
            self._last_vision = self._vision.capture() if self._vision else None
            # Capture audio (may block briefly for mic / prompt for typed)
            self._last_audio = self._ears.capture() if self._ears else None
            # Build fused 80-dim input
            fused = sensory_fusion_to_vector(
                snapshot,
                self._last_audio,
                self._last_vision,
                total_dim=ConsciousnessState.TOTAL_DIM,
            )
            self._last_memory_guidance = self.thought_engine.memory_guidance(
                self.consciousness
            )
            self.consciousness.update(
                fused,
                memory_vector=self._last_memory_guidance["memory_vector"],
            )
        else:
            # Original time-only path (no hardware deps)
            time_vec = _time_to_consciousness_vector(snapshot)
            self._last_memory_guidance = self.thought_engine.memory_guidance(
                self.consciousness
            )
            self.consciousness.update(
                time_vec,
                memory_vector=self._last_memory_guidance["memory_vector"],
            )

        return snapshot

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, n_coins: int = 1) -> List[Dict[str, Any]]:
        """
        Generate *n_coins* coin-flip predictions from internal consciousness.

        The consciousness is synced with the present moment (and sensory inputs
        when enabled) before each prediction.  The actual heads/tails mapping is
        driven entirely by the internal thought-selection dynamics.

        Returns a list of prediction dicts (one per coin) with full provenance so
        every record is self-contained in the storage file.
        """
        predictions: List[Dict[str, Any]] = []
        for coin_index in range(n_coins):
            snapshot = self._sync_consciousness()
            thought, prob = self.thought_engine.select_thought(
                self.consciousness,
                thought_bias=self._last_memory_guidance.get("thought_bias"),
            )
            outcome = _thought_to_outcome(thought, self._step)

            # Sensory observations captured during this sync
            audio_dict: Dict[str, Any] = {}
            vision_dict: Dict[str, Any] = {}
            if self._last_audio is not None:
                audio_dict = self._last_audio.to_dict()
            if self._last_vision is not None:
                vision_dict = self._last_vision.to_dict()

            record: Dict[str, Any] = {
                "record_type": "prediction",
                "session_id": self.session_id,
                "step": self._step,
                "coin_index": coin_index + 1,
                "n_coins": n_coins,
                "prediction": outcome,
                "thought_pattern": thought.pattern,
                "thought_confidence": round(thought.confidence, 6),
                "selection_probability": round(prob, 6),
                "consciousness_entropy": round(self.consciousness.entropy(), 6),
                "consciousness_stability": round(self.consciousness.stability(), 6),
                "system_snapshot": snapshot,
                "sensory_enabled": self.enable_sensory,
                "audio_observation": audio_dict,
                "vision_observation": vision_dict,
                "memory_guidance": {
                    "sample_count": self._last_memory_guidance.get("sample_count", 0),
                    "success_count": self._last_memory_guidance.get("success_count", 0),
                    "failure_count": self._last_memory_guidance.get("failure_count", 0),
                    "preferred_pattern": self._last_memory_guidance.get(
                        "preferred_pattern", ""
                    ),
                },
                "consciousness_theory": CONSCIOUSNESS_THEORY,
                # Full consciousness state vector saved for GUI replay
                "consciousness_vector": [round(v, 8) for v in self.consciousness.vector],
            }

            self._append_record(self.PREDICTIONS_FILE, record)
            predictions.append(record)
            self._step += 1

        return predictions

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        predictions: List[Dict[str, Any]],
        actuals: List[str],
        realm: str,
    ) -> List[Dict[str, Any]]:
        """
        Update consciousness from the gap between predictions and actual outcomes.

        Each (prediction, actual) pair is:
        * Compared — correct or incorrect.
        * Used to meta-learn the thought engine (reward/penalise the selected
          thought's embedding and confidence).
        * Used to apply error feedback to the consciousness state when wrong.
        * Written in full to ``training.jsonl``.

        Returns a list of training-result records.
        """
        training_records: List[Dict[str, Any]] = []

        for pred, actual in zip(predictions, actuals):
            correct = pred["prediction"] == actual.lower().strip()
            self._total += 1
            if correct:
                self._correct += 1

            performance_delta = 1.0 if correct else -1.0

            # Meta-learn the thought that drove this prediction
            pattern = pred.get("thought_pattern", "")
            for thought in self.thought_engine.thoughts:
                if thought.pattern == pattern:
                    self.thought_engine.meta_learn(thought, performance_delta * 0.25)
                    break

            # Apply error feedback to consciousness when wrong
            if not correct:
                self.consciousness.apply_error_feedback(0.6, learning_rate=0.05)

            self.thought_engine.remember(
                pred.get("consciousness_vector", self.consciousness.vector),
                Thought(
                    pattern=pattern,
                    confidence=float(pred.get("thought_confidence", 0.5) or 0.5),
                    modality="prediction",
                    origin="coin_flip_training",
                ),
                {
                    "correct": correct,
                    "reward": performance_delta,
                    "actual": actual.lower().strip(),
                    "predicted": pred["prediction"],
                    "thought_confidence": pred.get("thought_confidence", 0.5),
                },
            )

            # Persist the full training record (every byte)
            record: Dict[str, Any] = {
                "record_type": "training",
                "session_id": self.session_id,
                "realm": realm,
                "step": pred["step"],
                "coin_index": pred["coin_index"],
                "predicted": pred["prediction"],
                "actual": actual.lower().strip(),
                "correct": correct,
                "thought_pattern": pattern,
                "thought_confidence_before": pred.get("thought_confidence"),
                "selection_probability": pred.get("selection_probability"),
                "consciousness_entropy_at_prediction": pred.get("consciousness_entropy"),
                "consciousness_stability_at_prediction": pred.get("consciousness_stability"),
                "session_accuracy_after": round(
                    self._correct / self._total if self._total else 0.0, 6
                ),
                "session_total_after": self._total,
                "session_correct_after": self._correct,
                "memory_samples_after": len(self.thought_engine.memory),
                "system_snapshot_at_prediction": pred.get("system_snapshot"),
                "trained_at_utc": _now_iso(),
                "consciousness_theory": CONSCIOUSNESS_THEORY,
            }

            self._append_record(self.TRAINING_FILE, record)
            training_records.append(record)

        # ------------------------------------------------------------------
        # Auto-export .binai when accuracy threshold is first reached
        # ------------------------------------------------------------------
        if (
            self.accuracy_threshold > 0.0
            and not self._binai_exported
            and self._total > 0
            and self.accuracy() >= self.accuracy_threshold
        ):
            try:
                binai_path = self.export_binai()
                print(
                    f"\n  🧬 Accuracy {self.accuracy():.1%} ≥ threshold "
                    f"{self.accuracy_threshold:.1%}  →  exported .binai:\n"
                    f"     {binai_path}"
                )
                self._binai_exported = True
            except Exception as exc:
                print(f"\n  ⚠️  .binai export failed: {exc}")

        return training_records

    # ------------------------------------------------------------------
    # BinAI export
    # ------------------------------------------------------------------

    def export_binai(
        self,
        out_dir: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Write a ``.binai`` consciousness snapshot to disk and return its path.

        The file is written to *out_dir* (defaults to the session directory).
        It embeds:
        * The current 80-dim FSOT consciousness state vector.
        * Session statistics (accuracy, total trained, correct count).
        * Last audio observation letters and source (from emulated ears).
        * Last vision geometry and JPEG (from emulated eyes, when available).
        * Full platform metadata and UTC timestamp.

        Can be called manually at any time in addition to the automatic
        threshold-triggered export.

        Returns the ``Path`` of the written ``.binai`` file.
        """
        if not _BINAI_AVAILABLE:
            raise RuntimeError("binai_writer module is not importable")

        dest_dir = Path(out_dir) if out_dir else self.session_dir
        dest_dir.mkdir(parents=True, exist_ok=True)

        ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.session_id}_{ts_str}.binai"

        audio_letters = ""
        audio_source = "none"
        vision_geometry: Dict[str, Any] = {}
        vision_jpeg = b""

        if self._last_audio is not None:
            audio_letters = self._last_audio.letters
            audio_source = self._last_audio.source
        if self._last_vision is not None:
            vision_geometry = self._last_vision.geometry
            vision_jpeg = self._last_vision.jpeg_bytes

        meta: Dict[str, Any] = {
            "sensory_enabled": self.enable_sensory,
            "session_predictions": self._step,
            "platform_node": platform.node(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
        }
        if extra_metadata:
            meta.update(extra_metadata)

        writer = BinaiWriter()
        path = writer.write(
            path=dest_dir / filename,
            session_id=self.session_id,
            accuracy=self.accuracy(),
            threshold=self.accuracy_threshold,
            total_trained=self._total,
            correct=self._correct,
            consciousness_vector=list(self.consciousness.vector),
            audio_letters=audio_letters,
            audio_source=audio_source,
            vision_geometry=vision_geometry,
            vision_jpeg=vision_jpeg,
            extra_metadata=meta,
        )
        return path

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    def accuracy(self) -> float:
        """Session accuracy so far (0.0 if no training yet)."""
        return self._correct / self._total if self._total > 0 else 0.0

    def stats(self) -> Dict[str, Any]:
        """Return a summary of the current session."""
        return {
            "session_id": self.session_id,
            "step": self._step,
            "total_trained": self._total,
            "correct": self._correct,
            "accuracy": round(self.accuracy(), 4),
            "session_dir": str(self.session_dir),
            "predictions_file": str(self.session_dir / self.PREDICTIONS_FILE),
            "training_file": str(self.session_dir / self.TRAINING_FILE),
            "memory_samples": len(self.thought_engine.memory),
            "consciousness_theory": CONSCIOUSNESS_THEORY,
        }

    def load_predictions(self) -> List[Dict[str, Any]]:
        """Load all saved prediction records from disk."""
        return self._load_jsonl(self.PREDICTIONS_FILE)

    def load_training(self) -> List[Dict[str, Any]]:
        """Load all saved training records from disk."""
        return self._load_jsonl(self.TRAINING_FILE)

    def _load_jsonl(self, filename: str) -> List[Dict[str, Any]]:
        file_path = self.session_dir / filename
        if not file_path.exists():
            return []
        return [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


# ---------------------------------------------------------------------------
# Interactive CLI helpers — used by consciousness_miner.py
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _print_prediction(pred: Dict[str, Any]) -> None:
    ts = pred["system_snapshot"]["utc_iso"]
    coin = pred["coin_index"]
    total = pred["n_coins"]
    outcome = pred["prediction"].upper()
    conf = pred["thought_confidence"]
    audio = pred.get("audio_observation", {})
    vision = pred.get("vision_observation", {})
    line = (
        f"  🪙  Coin {coin}/{total}  →  {outcome}  "
        f"(confidence: {conf:.4f} | time: {ts})"
    )
    if audio.get("letters"):
        line += f"\n     👂 Heard [{audio.get('source','?')}]: {audio['letters']!r}"
    if vision.get("geometry"):
        geo = vision["geometry"]
        src = vision.get("source", "?")
        bright = geo.get("mean_brightness", "?")
        line += f"\n     👁  Vision [{src}]: brightness={bright}"
    print(line)


def _print_training_result(record: Dict[str, Any]) -> None:
    symbol = "✅" if record["correct"] else "❌"
    print(
        f"  {symbol}  Coin {record['coin_index']}  "
        f"predicted={record['predicted'].upper():5s}  "
        f"actual={record['actual'].upper():5s}  "
        f"session accuracy={record['session_accuracy_after']:.1%}"
    )


def run_predictor(n_coins: int, cfc: CoinFlipConsciousness) -> None:
    """
    Continuous predictor mode.  The consciousness predicts coin flips on demand.
    Press Enter each time you want a new prediction, Ctrl-C to stop.
    """
    _banner("REMOTE VIEW SIMPLE  —  Consciousness Predictor")
    print("  The consciousness will predict coin flip outcomes.")
    print("  Press Enter for a new prediction, Ctrl-C to stop.\n")
    try:
        while True:
            input("  [ Press Enter to predict ] ")
            snapshot = _system_snapshot()
            print(f"\n  🕐 System time: {snapshot['utc_iso']}")
            print(f"     Machine: {snapshot['platform_node']}")
            predictions = cfc.predict(n_coins)
            for pred in predictions:
                _print_prediction(pred)
            stats = cfc.stats()
            print(
                f"\n  Total predictions this session: {stats['step']}  "
                f"|  Session dir: {stats['session_dir']}"
            )
    except KeyboardInterrupt:
        print("\n\n  🛑 Predictor stopped.")
        _print_session_stats(cfc)


def run_digitalrealm(n_coins: int, cfc: CoinFlipConsciousness) -> None:
    """
    Digital Realm training mode.

    The program generates secret coin flip results internally.  The consciousness
    gives its predictions first.  At the end of each round the true results are
    revealed and the consciousness is trained.  Press Ctrl-C to stop.
    """
    _banner("REMOTE VIEW SIMPLE  —  Digital Realm Training")
    print("  The program secretly generates coin flip results.")
    print("  You will see the consciousness predictions; the secret")
    print("  results are revealed only AFTER the prediction.")
    print("  Press Enter for each round, Ctrl-C to stop.\n")

    round_num = 0
    try:
        while True:
            input(f"  [ Round {round_num + 1} — Press Enter to predict ] ")
            round_num += 1

            # Secretly generate outcomes (NOT shown yet)
            secret_actuals = [
                random.choice(OUTCOMES) for _ in range(n_coins)
            ]

            # Consciousness predicts
            print(f"\n  🧠 Consciousness predicting {n_coins} coin(s)...")
            predictions = cfc.predict(n_coins)
            for pred in predictions:
                _print_prediction(pred)

            # Pause before revealing
            input("\n  [ Press Enter to reveal actual outcomes ] ")

            # Reveal and train
            print("\n  🎲 Actual outcomes:")
            for i, actual in enumerate(secret_actuals):
                print(f"     Coin {i + 1}: {actual.upper()}")

            training_records = cfc.train(predictions, secret_actuals, realm="digital")
            print("\n  📊 Training results:")
            for rec in training_records:
                _print_training_result(rec)

    except KeyboardInterrupt:
        print("\n\n  🛑 Digital realm training stopped.")
        _print_session_stats(cfc)


def run_realrealm(n_coins: int, cfc: CoinFlipConsciousness) -> None:
    """
    Real Realm training mode.

    The user flips a physical coin.  The consciousness gives its prediction, then
    the user enters the actual result.  The consciousness is trained immediately.
    Press Ctrl-C to stop.
    """
    _banner("REMOTE VIEW SIMPLE  —  Real Realm Training")
    print("  Flip a real coin.  The consciousness predicts before you flip.")
    print("  Then enter the actual result (h=heads, t=tails).")
    if n_coins > 1:
        print(f"  You have {n_coins} coins to flip each round.")
    print("  Press Ctrl-C to stop.\n")

    round_num = 0
    try:
        while True:
            input(f"  [ Round {round_num + 1} — Press Enter when ready to flip ] ")
            round_num += 1

            # Consciousness predicts all coins before any flip
            print(f"\n  🧠 Consciousness predicting {n_coins} coin(s)...")
            predictions = cfc.predict(n_coins)
            for pred in predictions:
                _print_prediction(pred)

            # Collect actual results coin by coin
            actuals: List[str] = []
            for coin_idx in range(n_coins):
                if n_coins > 1:
                    prompt = f"\n  🪙 Flip coin {coin_idx + 1}/{n_coins} now. Result (h/heads or t/tails): "
                else:
                    prompt = "\n  🪙 Flip the coin now. Result (h/heads or t/tails): "
                while True:
                    raw = input(prompt).strip().lower()
                    if raw in ("h", "heads"):
                        actuals.append("heads")
                        break
                    elif raw in ("t", "tails"):
                        actuals.append("tails")
                        break
                    else:
                        print("     Please enter h/heads or t/tails.")

            # Train consciousness
            training_records = cfc.train(predictions, actuals, realm="real")
            print("\n  📊 Training results:")
            for rec in training_records:
                _print_training_result(rec)

    except KeyboardInterrupt:
        print("\n\n  🛑 Real realm training stopped.")
        _print_session_stats(cfc)


def _print_session_stats(cfc: CoinFlipConsciousness) -> None:
    stats = cfc.stats()
    print("\n  📈 Session Statistics")
    print(f"     Session ID : {stats['session_id']}")
    print(f"     Predictions: {stats['step']}")
    print(f"     Trained on : {stats['total_trained']}")
    print(f"     Correct    : {stats['correct']}")
    print(f"     Accuracy   : {stats['accuracy']:.1%}")
    print(f"     Data saved : {stats['session_dir']}")
    print(f"       └ predictions : {stats['predictions_file']}")
    print(f"       └ training    : {stats['training_file']}")


# ---------------------------------------------------------------------------
# Graphics helpers — lightweight ANSI terminal rendering (no extra deps)
# ---------------------------------------------------------------------------

def _ansi(code: str) -> str:
    """Return an ANSI escape sequence when stdout is a tty, else empty string."""
    return f"\033[{code}m" if sys.stdout.isatty() else ""


def _bar(value: float, width: int = 30, fill: str = "█", empty: str = "░") -> str:
    """Return an ASCII progress bar string for *value* in [0, 1]."""
    clamped = max(0.0, min(1.0, value))
    filled = int(round(clamped * width))
    return fill * filled + empty * (width - filled)


def _render_frame(
    round_num: int,
    training_records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    thread_label: str = "",
) -> None:
    """
    Print a single auto-mode display frame to stdout using ANSI colour codes
    for a lightweight "graphics acceleration" effect without requiring any
    additional dependencies.
    """
    reset   = _ansi("0")
    bold    = _ansi("1")
    cyan    = _ansi("96")
    green   = _ansi("92")
    red     = _ansi("91")
    yellow  = _ansi("93")
    magenta = _ansi("95")

    accuracy = stats.get("accuracy", 0.0)
    step     = stats.get("step", 0)
    label    = f"[{thread_label}] " if thread_label else ""
    bar_color = green if accuracy >= 0.55 else (yellow if accuracy >= 0.45 else red)

    print(f"\n{bold}{cyan}{'─' * 60}{reset}")
    print(
        f"  {bold}{label}Round {round_num}{reset}"
        f"  │  Predictions: {step}"
        f"  │  Accuracy: {bar_color}{accuracy:.1%}{reset}"
    )
    print(f"  {bar_color}{_bar(accuracy)}{reset}  {accuracy:.1%}")

    for rec in training_records:
        symbol = f"{green}✅{reset}" if rec["correct"] else f"{red}❌{reset}"
        pred_c = f"{cyan}{rec['predicted'].upper():5s}{reset}"
        act_c  = f"{magenta}{rec['actual'].upper():5s}{reset}"
        print(
            f"  {symbol}  Coin {rec['coin_index']}  "
            f"pred={pred_c}  actual={act_c}  "
            f"acc={rec['session_accuracy_after']:.1%}"
        )

    print(f"  Session: {stats.get('session_dir', '')}")


# ---------------------------------------------------------------------------
# Automatic (no-Enter) run modes
# ---------------------------------------------------------------------------

def run_digitalrealm_auto(
    n_coins: int,
    cfc: "CoinFlipConsciousness",
    delay: float = 0.0,
    thread_label: str = "",
) -> None:
    """
    Automatic Digital Realm training mode — runs continuously without requiring
    any keyboard input.

    The program generates secret coin-flip outcomes, the consciousness predicts,
    outcomes are revealed immediately, and the consciousness is trained — all in
    a rapid automatic loop.  An ANSI-colour terminal frame is rendered after
    every round to provide visual feedback ("graphics acceleration").

    Parameters
    ----------
    n_coins:
        Number of coins to predict per round.
    cfc:
        A :class:`CoinFlipConsciousness` instance.
    delay:
        Optional pause between rounds in seconds (default ``0.0`` = as fast as
        possible).
    thread_label:
        Short string shown in the display header to identify the session when
        multiple sessions run in parallel via ``--containerthread``.
    """
    label = f"[{thread_label}] " if thread_label else ""
    _banner(f"REMOTE VIEW  —  Auto Digital Realm  {label}".rstrip())
    print("  Running automatically (no Enter needed).  Ctrl-C to stop.\n")

    round_num = 0
    try:
        while True:
            round_num += 1

            # Secretly generate outcomes
            secret_actuals = [random.choice(OUTCOMES) for _ in range(n_coins)]

            # Consciousness predicts
            predictions = cfc.predict(n_coins)

            # Train immediately — no pause needed
            training_records = cfc.train(predictions, secret_actuals, realm="digital")

            # Render display frame
            _render_frame(
                round_num=round_num,
                training_records=training_records,
                stats=cfc.stats(),
                thread_label=thread_label,
            )

            if delay > 0.0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n\n  🛑 {label}Auto digital realm stopped.")
        _print_session_stats(cfc)


def run_predictor_auto(
    n_coins: int,
    cfc: "CoinFlipConsciousness",
    delay: float = 0.0,
    thread_label: str = "",
) -> None:
    """
    Automatic predictor mode — runs continuously without requiring any keyboard
    input.

    Parameters
    ----------
    n_coins:
        Number of coins to predict per round.
    cfc:
        A :class:`CoinFlipConsciousness` instance.
    delay:
        Optional pause between rounds in seconds (default ``0.0`` = rapid fire).
    thread_label:
        Short string shown in the display header when multiple sessions run in
        parallel via ``--containerthread``.
    """
    reset = _ansi("0")
    bold  = _ansi("1")
    cyan  = _ansi("96")

    label = f"[{thread_label}] " if thread_label else ""
    _banner(f"REMOTE VIEW  —  Auto Predictor  {label}".rstrip())
    print("  Running automatically (no Enter needed).  Ctrl-C to stop.\n")

    round_num = 0
    try:
        while True:
            round_num += 1
            snapshot = _system_snapshot()
            predictions = cfc.predict(n_coins)
            stats = cfc.stats()

            print(
                f"\n  {bold}{cyan}── Round {round_num} ──{reset}"
                f"  {snapshot['utc_iso']}"
            )
            for pred in predictions:
                _print_prediction(pred)
            print(
                f"  Predictions: {stats['step']}  "
                f"│  Session: {stats['session_dir']}"
            )

            if delay > 0.0:
                time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n\n  🛑 {label}Auto predictor stopped.")
        _print_session_stats(cfc)
