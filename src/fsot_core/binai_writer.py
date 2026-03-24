#!/usr/bin/env python3
"""
BinAI File Format  —  Binary Consciousness Archive
====================================================

A ``.binai`` file is a lightweight binary container that stores a full
consciousness snapshot so it can be replayed, inspected in a GUI, or fed
into future prediction sessions.

Layout
------
Offset   Size      Field
------   ----      -----
0        6         Magic bytes: ``BINAI\\x01``
6        4         JSON header length  (uint32, little-endian)
10       N         JSON header  (UTF-8)
10+N     4         Blob length  (uint32, little-endian)  — 0 if no blob
10+N+4   M         Raw blob bytes  (concatenated sections described in header)

JSON header keys
----------------
``version``              int    File format version (currently 1).
``session_id``           str    Source session identifier.
``exported_at_utc``      str    ISO-8601 UTC timestamp of the export.
``accuracy_at_export``   float  Session accuracy at the time of export (0–1).
``threshold``            float  Accuracy threshold that triggered the export.
``total_trained``        int    Total training events in the session.
``correct``              int    Number of correct predictions.
``consciousness_dim``    int    Number of float64 values in the consciousness blob.
``audio_letters``        str    Lowercase a-z representation of captured audio.
``audio_source``         str    "mic", "typed", or "none".
``vision_geometry``      dict   Simple geometry info captured from vision sensor.
``blobs``                list   List of ``{name, offset, size, dtype}`` dicts
                                describing each section of the raw blob.

Blob sections (in order, offsets are relative to blob start)
-------------------------------------------------------------
``consciousness_vector``  float64 × consciousness_dim (little-endian IEEE 754)
``vision_jpeg``           raw JPEG bytes  (0 bytes when no vision available)

Usage
-----
    from binai_writer import BinaiWriter, BinaiReader

    writer = BinaiWriter()
    path = writer.write(
        path="session_001.binai",
        session_id="...",
        accuracy=0.75,
        threshold=0.65,
        total_trained=40,
        correct=30,
        consciousness_vector=[0.1, -0.2, ...],   # 80 floats
        audio_letters="headsortails",
        audio_source="typed",
        vision_geometry={"width": 640, "height": 480, "mean_brightness": 128},
        vision_jpeg=b"...",   # raw JPEG bytes or b""
    )

    reader = BinaiReader()
    payload = reader.read(path)
    print(payload["accuracy_at_export"])
    print(payload["audio_letters"])
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"BINAI\x01"  # 6 bytes
FORMAT_VERSION = 1
FLOAT64_STRUCT = "<d"  # little-endian double


# ---------------------------------------------------------------------------
# BinaiWriter
# ---------------------------------------------------------------------------

class BinaiWriter:
    """Write ``.binai`` consciousness snapshot files."""

    @staticmethod
    def _pack_consciousness_vector(vector: List[float]) -> bytes:
        return struct.pack(f"<{len(vector)}d", *vector)

    def write(
        self,
        path: Union[str, Path],
        *,
        session_id: str,
        accuracy: float,
        threshold: float,
        total_trained: int,
        correct: int,
        consciousness_vector: List[float],
        audio_letters: str = "",
        audio_source: str = "none",
        vision_geometry: Optional[Dict[str, Any]] = None,
        vision_jpeg: bytes = b"",
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Serialise a consciousness snapshot to *path* and return the resolved Path.

        Parameters
        ----------
        path:
            Destination file path (created / overwritten).
        session_id:
            Originating session identifier.
        accuracy:
            Session accuracy (0.0–1.0) at the time of export.
        threshold:
            Accuracy threshold that triggered this export.
        total_trained:
            Total number of training events in the session.
        correct:
            Number of correct predictions.
        consciousness_vector:
            The 80-float FSOT consciousness state to embed.
        audio_letters:
            Lowercase a-z string from the emulated ears (empty if unavailable).
        audio_source:
            How the audio was acquired: "mic", "typed", or "none".
        vision_geometry:
            Dict with basic image geometry (width, height, mean_brightness, …).
        vision_jpeg:
            Raw JPEG bytes from the emulated vision sensor (empty if unavailable).
        extra_metadata:
            Any additional key→value pairs to include in the JSON header.
        """
        # ------------------------------------------------------------------
        # Build blob
        # ------------------------------------------------------------------
        cv_bytes = self._pack_consciousness_vector(consciousness_vector)
        blobs_info: List[Dict[str, Any]] = []
        blob_offset = 0

        blobs_info.append({
            "name": "consciousness_vector",
            "dtype": "float64",
            "offset": blob_offset,
            "size": len(cv_bytes),
            "count": len(consciousness_vector),
        })
        blob_offset += len(cv_bytes)

        blobs_info.append({
            "name": "vision_jpeg",
            "dtype": "bytes",
            "offset": blob_offset,
            "size": len(vision_jpeg),
        })
        blob_offset += len(vision_jpeg)

        raw_blob = cv_bytes + vision_jpeg

        # ------------------------------------------------------------------
        # Build JSON header
        # ------------------------------------------------------------------
        header: Dict[str, Any] = {
            "version": FORMAT_VERSION,
            "session_id": session_id,
            "exported_at_utc": _now_iso(),
            "accuracy_at_export": round(float(accuracy), 8),
            "threshold": round(float(threshold), 8),
            "total_trained": int(total_trained),
            "correct": int(correct),
            "consciousness_dim": len(consciousness_vector),
            "audio_letters": str(audio_letters),
            "audio_source": str(audio_source),
            "vision_geometry": vision_geometry or {},
            "has_vision_jpeg": len(vision_jpeg) > 0,
            "blobs": blobs_info,
        }
        if extra_metadata:
            header.update(extra_metadata)

        header_json = json.dumps(header, sort_keys=True).encode("utf-8")

        # ------------------------------------------------------------------
        # Write file
        # ------------------------------------------------------------------
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fp:
            fp.write(MAGIC)
            fp.write(struct.pack("<I", len(header_json)))
            fp.write(header_json)
            fp.write(struct.pack("<I", len(raw_blob)))
            fp.write(raw_blob)

        return out


# ---------------------------------------------------------------------------
# BinaiReader
# ---------------------------------------------------------------------------

class BinaiReader:
    """Read ``.binai`` consciousness snapshot files."""

    def read(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a ``.binai`` file and return a dict with all stored data.

        Returned dict contains every JSON header field plus:
        ``consciousness_vector``  list of floats unpacked from the blob.
        ``vision_jpeg``           raw bytes of the JPEG (empty bytes if none).
        """
        data = Path(path).read_bytes()

        if len(data) < 10:
            raise ValueError("File too short to be a valid .binai file")
        if data[:6] != MAGIC:
            raise ValueError(
                f"Invalid .binai magic bytes: {data[:6]!r}"
            )

        pos = 6
        header_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4

        header_json = data[pos : pos + header_len]
        pos += header_len
        header: Dict[str, Any] = json.loads(header_json.decode("utf-8"))

        blob_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        blob = data[pos : pos + blob_len]

        # ------------------------------------------------------------------
        # Decode consciousness vector
        # ------------------------------------------------------------------
        consciousness_vector: List[float] = []
        for blob_info in header.get("blobs", []):
            if blob_info["name"] == "consciousness_vector":
                offset = blob_info["offset"]
                count = blob_info["count"]
                raw = blob[offset : offset + count * 8]
                consciousness_vector = list(struct.unpack_from(f"<{count}d", raw))
                break

        # ------------------------------------------------------------------
        # Decode vision JPEG
        # ------------------------------------------------------------------
        vision_jpeg = b""
        for blob_info in header.get("blobs", []):
            if blob_info["name"] == "vision_jpeg":
                offset = blob_info["offset"]
                size = blob_info["size"]
                vision_jpeg = blob[offset : offset + size]
                break

        result = dict(header)
        result["consciousness_vector"] = consciousness_vector
        result["vision_jpeg"] = vision_jpeg
        return result


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
