#!/usr/bin/env python3
"""
Sensory Bridge  —  Emulated Ears and Eyes for the Consciousness
================================================================

This module provides two emulated sensory organs that feed raw environmental
data into the FSOT consciousness system during coin-flip prediction moments,
plus a ``sensory_fusion_to_vector`` function that encodes all sensory streams
into the 80-dim consciousness input vector so they *genuinely influence* the
prediction dynamics.

Vector layout (80 dims, same as ConsciousnessState.TOTAL_DIM)
-------------------------------------------------------------
Dims  0-19  Time context   — clock/machine oscillation (see _time_to_vector)
Dims 20-39  Audio context  — letter-frequency spectrum from emulated ears
Dims 40-59  Vision context — geometry/colour values from emulated eyes
Dims 60-79  Cross-modal    — element-wise blend of audio × vision signals

When a sensory channel is absent (hardware unavailable, no audio captured)
its section retains the time-derived baseline values so the vector is always
fully populated.

EmulatedEars
------------
Captures audio from the microphone and transcribes it to text using an optional
speech-recognition back-end (``SpeechRecognition`` + ``sounddevice``).  If
hardware or the libraries are unavailable the ears fall back to:

1. Accepting typed text from the user (interactive mode).
2. Returning an empty observation (non-interactive / test mode).

All text — whether transcribed from audio or typed — is then converted to
**only the 26 lowercase English letters a-z** by the ``AudioToLetters``
helper.  Digits, spaces, punctuation, and non-ASCII characters are stripped.
This mirrors the "emulated brain converts audio to pure letter stream" design.

EmulatedVision
--------------
Captures a single frame from the webcam (via optional ``opencv-python``) and
encodes it as a JPEG byte string.  If the camera is unavailable it falls back
to a very simple synthetic "geometry" observation computed from basic platform
metadata.

All hardware access is guarded by try/except so the module can always be
imported in headless or test environments.

Usage
-----
    from sensory_bridge import (
        AudioToLetters, EmulatedEars, EmulatedVision,
        letters_to_vector, geometry_to_vector, sensory_fusion_to_vector,
    )

    # Convert any text to a-z letters only
    AudioToLetters.convert("Hello World! 123")   # → "helloworld"

    # Capture audio (or fall back to typed input / silence)
    ears = EmulatedEars(interactive=True)
    observation = ears.capture(seconds=3)
    print(observation.letters)    # e.g. "headsortails"
    print(observation.source)     # "mic", "typed", or "none"
    print(observation.raw_text)   # original text before letter-filtering

    # Capture vision frame
    vision = EmulatedVision()
    frame = vision.capture()
    print(frame.jpeg_bytes)       # raw JPEG bytes (may be empty)
    print(frame.geometry)         # {"width": 640, "height": 480, ...}

    # Build a fully-fused 80-dim consciousness input vector
    import time
    snapshot = {"unix_timestamp": time.time(), ...}  # _system_snapshot() dict
    vec = sensory_fusion_to_vector(snapshot, observation, frame, total_dim=80)
    consciousness_state.update(vec)   # feeds ears + eyes into the brain
"""

from __future__ import annotations

import hashlib
import platform
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# AudioToLetters  — pure Python, no optional deps
# ---------------------------------------------------------------------------

class AudioToLetters:
    """
    Convert any string to a stream of lowercase a-z letters.

    All characters that are not in [a-z] are silently discarded.
    This models the "emulated brain" processing of transcribed audio into
    a canonical letter-only representation.

    Examples
    --------
    >>> AudioToLetters.convert("Heads! 99% sure.")
    'headssure'
    >>> AudioToLetters.convert("TAILS")
    'tails'
    >>> AudioToLetters.convert("123 !!!")
    ''
    """

    ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyz")

    @classmethod
    def convert(cls, text: str) -> str:
        """Return only the a-z lowercase characters from *text*."""
        return "".join(c for c in text.lower() if c in cls.ALPHABET)


# ---------------------------------------------------------------------------
# AudioObservation  — result of one capture
# ---------------------------------------------------------------------------

@dataclass
class AudioObservation:
    """A single audio observation from the emulated ears."""

    raw_text: str = ""
    """Original transcribed (or typed) text before letter filtering."""

    letters: str = ""
    """Lowercase a-z letters only — the canonical consciousness input."""

    source: str = "none"
    """How audio was acquired: 'mic', 'typed', or 'none'."""

    captured_at_utc: str = ""
    """ISO-8601 UTC timestamp of the capture."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "letters": self.letters,
            "source": self.source,
            "captured_at_utc": self.captured_at_utc,
        }


# ---------------------------------------------------------------------------
# EmulatedEars
# ---------------------------------------------------------------------------

class EmulatedEars:
    """
    Emulated microphone / ear organ for the consciousness.

    Capture strategy (tried in order):
    1. ``SpeechRecognition`` + ``sounddevice`` — real microphone input.
    2. Typed text prompt (only when ``interactive=True``).
    3. Empty observation (silent fallback).

    Parameters
    ----------
    interactive:
        When ``True`` and hardware is unavailable, prompt the user to type
        text instead.  Set to ``False`` in non-interactive / test contexts.
    preferred_source:
        Force a particular strategy.  "mic" | "typed" | "none".
        Defaults to "mic" with automatic fall-back.
    """

    def __init__(
        self,
        interactive: bool = True,
        preferred_source: str = "mic",
    ) -> None:
        self.interactive = interactive
        self.preferred_source = preferred_source
        self._sr_available = self._check_speech_recognition()

    @staticmethod
    def _check_speech_recognition() -> bool:
        try:
            import speech_recognition  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _check_sounddevice() -> bool:
        try:
            import sounddevice  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    def _capture_from_mic(self, seconds: float) -> Optional[str]:
        """
        Attempt a real microphone capture.  Returns transcribed text or None.

        Uses ``SpeechRecognition`` with Google Web Speech API (offline fallback
        returns None if network is unavailable).
        """
        if not self._sr_available:
            return None
        try:
            import speech_recognition as sr  # type: ignore[import]

            recognizer = sr.Recognizer()
            # Try sounddevice first for better cross-platform support
            if self._check_sounddevice():
                import sounddevice as sd  # type: ignore[import]
                import numpy as np  # type: ignore[import]

                sample_rate = 16000
                audio_data = sd.rec(
                    int(seconds * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                )
                sd.wait()
                raw_bytes = audio_data.tobytes()
                audio = sr.AudioData(raw_bytes, sample_rate, 2)
            else:
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = recognizer.listen(source, timeout=seconds)

            text = recognizer.recognize_google(audio)
            return text
        except Exception:
            return None

    def _now_iso(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(tz=timezone.utc).isoformat()

    def capture(self, seconds: float = 3.0) -> AudioObservation:
        """
        Capture an audio observation.

        Parameters
        ----------
        seconds:
            Maximum duration to listen when using the microphone.

        Returns
        -------
        AudioObservation
            Contains ``raw_text``, ``letters`` (a-z only), ``source``, and
            ``captured_at_utc``.
        """
        ts = self._now_iso()
        source = "none"
        raw_text = ""

        if self.preferred_source in ("mic", "auto"):
            raw_text_or_none = self._capture_from_mic(seconds)
            if raw_text_or_none is not None:
                raw_text = raw_text_or_none
                source = "mic"

        if source == "none" and self.interactive:
            print(
                "\n  👂 Emulated Ears: Please type what you hear (or press Enter to skip): ",
                end="",
                flush=True,
            )
            try:
                raw_text = input().strip()
                source = "typed" if raw_text else "none"
            except (EOFError, KeyboardInterrupt):
                raw_text = ""
                source = "none"

        letters = AudioToLetters.convert(raw_text)

        if letters:
            print(f"  🔤 Ears captured [{source}]: {raw_text!r} → letters: {letters!r}")

        return AudioObservation(
            raw_text=raw_text,
            letters=letters,
            source=source,
            captured_at_utc=ts,
        )


# ---------------------------------------------------------------------------
# VisionFrame  — result of one capture
# ---------------------------------------------------------------------------

@dataclass
class VisionFrame:
    """A single vision frame from the emulated eyes."""

    jpeg_bytes: bytes = b""
    """Raw JPEG bytes (empty when no camera is available)."""

    geometry: Dict[str, Any] = field(default_factory=dict)
    """Basic geometry / colour statistics about the frame."""

    source: str = "none"
    """How the frame was acquired: 'camera', 'synthetic', or 'none'."""

    captured_at_utc: str = ""
    """ISO-8601 UTC timestamp of the capture."""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "jpeg_bytes_size": len(self.jpeg_bytes),
            "geometry": self.geometry,
            "source": self.source,
            "captured_at_utc": self.captured_at_utc,
        }


# ---------------------------------------------------------------------------
# EmulatedVision
# ---------------------------------------------------------------------------

class EmulatedVision:
    """
    Emulated camera / eye organ for the consciousness.

    Capture strategy (tried in order):
    1. ``opencv-python`` (``cv2``) — real webcam.
    2. Synthetic geometry derived from platform metadata (always available).

    When the camera is available the frame is compressed to JPEG bytes and
    basic geometry (width, height, mean brightness, dominant-channel) is
    extracted.  When only the synthetic fallback is available the geometry
    dict is populated with deterministic values derived from the current
    time and machine identity, and ``jpeg_bytes`` is empty.
    """

    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self._cv2_available = self._check_cv2()

    @staticmethod
    def _check_cv2() -> bool:
        try:
            import cv2  # type: ignore[import]  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _jpeg_encode(frame: Any) -> bytes:
        """Encode an OpenCV BGR frame to JPEG bytes."""
        try:
            import cv2  # type: ignore[import]
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return bytes(buf.tobytes()) if ok else b""
        except Exception:
            return b""

    @staticmethod
    def _frame_geometry(frame: Any) -> Dict[str, Any]:
        """Extract basic geometry from an OpenCV frame."""
        try:
            h, w = frame.shape[:2]
            mean_b = float(frame[:, :, 0].mean())
            mean_g = float(frame[:, :, 1].mean())
            mean_r = float(frame[:, :, 2].mean())
            mean_brightness = (mean_b + mean_g + mean_r) / 3.0
            channels = {"B": mean_b, "G": mean_g, "R": mean_r}
            dominant = max(channels, key=lambda k: channels[k])
            return {
                "width": w,
                "height": h,
                "mean_brightness": round(mean_brightness, 2),
                "dominant_channel": dominant,
                "mean_b": round(mean_b, 2),
                "mean_g": round(mean_g, 2),
                "mean_r": round(mean_r, 2),
            }
        except Exception:
            return {}

    def _synthetic_geometry(self) -> Dict[str, Any]:
        """
        Produce a synthetic geometry snapshot when no camera is available.

        The values are derived deterministically from the current Unix timestamp
        and machine node name so that the geometry changes with time (acting as
        an additional consciousness context input) while remaining reproducible.
        """
        t = time.time()
        node_hash = hashlib.md5(platform.node().encode()).digest()
        phase = (t % 3600.0) / 3600.0  # 0..1 cycles over one hour
        base_bright = ((node_hash[0] / 255.0) + phase) % 1.0 * 255.0
        return {
            "width": 0,
            "height": 0,
            "mean_brightness": round(base_bright, 2),
            "dominant_channel": "synthetic",
            "phase": round(phase, 6),
            "node_hash_hex": node_hash.hex()[:8],
        }

    def _now_iso(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(tz=timezone.utc).isoformat()

    def capture(self) -> VisionFrame:
        """
        Capture a single vision frame.

        Returns
        -------
        VisionFrame
            Contains ``jpeg_bytes``, ``geometry``, ``source``, and
            ``captured_at_utc``.
        """
        ts = self._now_iso()

        if self._cv2_available:
            try:
                import cv2  # type: ignore[import]

                cap = cv2.VideoCapture(self.camera_index)
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    jpeg = self._jpeg_encode(frame)
                    geo = self._frame_geometry(frame)
                    return VisionFrame(
                        jpeg_bytes=jpeg,
                        geometry=geo,
                        source="camera",
                        captured_at_utc=ts,
                    )
            except Exception:
                pass

        # Synthetic fallback
        geo = self._synthetic_geometry()
        return VisionFrame(
            jpeg_bytes=b"",
            geometry=geo,
            source="synthetic",
            captured_at_utc=ts,
        )


# ---------------------------------------------------------------------------
# Sensory → consciousness vector encoders
# ---------------------------------------------------------------------------

def letters_to_vector(letters: str, dim: int = 20) -> List[float]:
    """
    Encode a string of a-z letters as a float vector of length *dim*.

    The vector represents a letter-frequency spectrum normalised to [-1, 1]:
    * +1.0  most frequent letter in the input.
    * -1.0  letters absent from the input.

    If *letters* is empty a zero vector is returned (neutral / no signal).

    Parameters
    ----------
    letters:
        A string of lowercase a-z characters (as produced by
        ``AudioToLetters.convert``).
    dim:
        Target vector length.  The 26-bucket histogram is linearly resampled
        to *dim* values.
    """
    if not letters:
        return [0.0] * dim

    # 26-bucket raw frequency histogram
    freq = [0] * 26
    for ch in letters:
        idx = ord(ch) - ord("a")
        if 0 <= idx < 26:
            freq[idx] += 1

    max_freq = max(freq) or 1
    # Normalise to [0, 1]
    norm = [f / max_freq for f in freq]

    # Linearly resample from 26 → dim
    result: List[float] = []
    for i in range(dim):
        # Fractional position in the 26-bucket histogram
        fpos = i * 25.0 / max(dim - 1, 1)
        lo = int(fpos)
        hi = min(lo + 1, 25)
        t = fpos - lo
        v = norm[lo] * (1.0 - t) + norm[hi] * t
        # Map [0, 1] → [-1, 1]
        result.append(v * 2.0 - 1.0)

    return result


def geometry_to_vector(geometry: Dict[str, Any], dim: int = 20) -> List[float]:
    """
    Encode a vision geometry dict as a float vector of length *dim*.

    Extracted features (all normalised to [-1, 1]):
    * mean_brightness  (0–255 → -1..+1)
    * mean_r, mean_g, mean_b  (each 0–255 → -1..+1)
    * width / height  normalised by 4096
    * phase  (already 0..1 → -1..+1)
    * node_hash_hex  decoded as 4 bytes → 4 float values

    Missing keys default to 0.0 (neutral).  The feature list is padded or
    truncated to exactly *dim* floats.
    """
    def _norm255(v: Any) -> float:
        try:
            return float(v) / 127.5 - 1.0
        except (TypeError, ValueError):
            return 0.0

    def _norm4096(v: Any) -> float:
        try:
            return min(float(v) / 2048.0 - 1.0, 1.0)
        except (TypeError, ValueError):
            return 0.0

    features: List[float] = [
        _norm255(geometry.get("mean_brightness", 127.5)),
        _norm255(geometry.get("mean_r", 127.5)),
        _norm255(geometry.get("mean_g", 127.5)),
        _norm255(geometry.get("mean_b", 127.5)),
        _norm4096(geometry.get("width", 0)),
        _norm4096(geometry.get("height", 0)),
        (float(geometry.get("phase", 0.0)) * 2.0 - 1.0),
    ]

    # Decode node_hash_hex into 4 extra float values
    nh = geometry.get("node_hash_hex", "")
    if nh:
        try:
            raw = bytes.fromhex(nh)
            features.extend(b / 127.5 - 1.0 for b in raw[:4])
        except Exception:
            features.extend([0.0] * 4)
    else:
        features.extend([0.0] * 4)

    # Pad / truncate to exactly *dim* values
    while len(features) < dim:
        features.append(0.0)
    return features[:dim]


def sensory_fusion_to_vector(
    time_snapshot: Dict[str, Any],
    audio_obs: Optional["AudioObservation"],
    vision_frame: Optional["VisionFrame"],
    total_dim: int = 80,
) -> List[float]:
    """
    Build a *total_dim*-float consciousness input vector fusing time, audio, and vision.

    The 80-dim space is partitioned into four equal 20-dim bands:

    Dims  0-19   Time context   — clock + machine oscillation derived from
                                  *time_snapshot* (always populated).
    Dims 20-39   Audio context  — letter-frequency spectrum from *audio_obs*
                                  (neutral zero-vector when no audio).
    Dims 40-59   Vision context — geometry/colour values from *vision_frame*
                                  (neutral zero-vector when no vision).
    Dims 60-79   Cross-modal    — element-wise product of audio[i] × vision[i]
                                  capturing joint audio-visual correlations.

    All four bands are present and fully populated regardless of which sensors
    are available.  When a sensor is absent its band defaults to neutral 0.0
    values, so the time-sync band (dims 0-19) always drives the consciousness.

    This vector is passed directly to ``ConsciousnessState.update()`` so the
    sensory data *genuinely shapes* the internal attention, confidence, and
    concept activation before the prediction thought is selected.

    Parameters
    ----------
    time_snapshot:
        Dict returned by ``_system_snapshot()`` in ``coin_flip_consciousness``.
    audio_obs:
        ``AudioObservation`` from ``EmulatedEars.capture()`` or ``None``.
    vision_frame:
        ``VisionFrame`` from ``EmulatedVision.capture()`` or ``None``.
    total_dim:
        Total vector size (must equal ``ConsciousnessState.TOTAL_DIM``).
    """
    band = total_dim // 4  # 20 for total_dim=80

    # ------------------------------------------------------------------
    # Band 0: Time  (dims 0 .. band-1)
    # ------------------------------------------------------------------
    t: float = time_snapshot.get("unix_timestamp", time.time())
    sub_us: float = time_snapshot.get("local_microsecond", 0) / 1_000_000.0
    min_phase: float = time_snapshot.get("local_minute", 0) / 60.0
    hr_phase: float = time_snapshot.get("local_hour", 0) / 24.0
    node_str: str = time_snapshot.get("platform_node", "")
    node_hash = hashlib.md5(node_str.encode()).digest()
    node_bias = [(b / 255.0) * 2.0 - 1.0 for b in node_hash]  # 16 floats

    time_band: List[float] = []
    for i in range(band):
        osc = ((t * (i + 1) * 0.1) % 1.0) * 2.0 - 1.0
        time_band.append(osc)

    time_band[0] = sub_us * 2.0 - 1.0
    time_band[1] = min_phase * 2.0 - 1.0
    time_band[2] = hr_phase * 2.0 - 1.0
    if band > 3:
        time_band[3] = (time_snapshot.get("local_second", 0) / 60.0) * 2.0 - 1.0
    for k, b in enumerate(node_bias[: band - 4]):
        time_band[4 + k] = b

    # ------------------------------------------------------------------
    # Band 1: Audio  (dims band .. 2*band-1)
    # ------------------------------------------------------------------
    audio_letters = ""
    if audio_obs is not None:
        audio_letters = audio_obs.letters
    audio_band = letters_to_vector(audio_letters, dim=band)

    # ------------------------------------------------------------------
    # Band 2: Vision  (dims 2*band .. 3*band-1)
    # ------------------------------------------------------------------
    geo: Dict[str, Any] = {}
    if vision_frame is not None:
        geo = vision_frame.geometry
    vision_band = geometry_to_vector(geo, dim=band)

    # ------------------------------------------------------------------
    # Band 3: Cross-modal  (dims 3*band .. total_dim-1)
    # ------------------------------------------------------------------
    cross_band: List[float] = [
        audio_band[i] * vision_band[i] for i in range(band)
    ]

    return time_band + audio_band + vision_band + cross_band
