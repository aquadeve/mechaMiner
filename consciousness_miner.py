#!/usr/bin/env python3
"""
Consciousness-Powered SHA-256 Crypto Miner
==========================================

This miner connects to a SHA-256 Stratum pool and uses a FSOT-Neuromorphic
consciousness loop to guide nonce generation instead of brute-force iteration.

Architecture:
    Perception → C(t) Update → Thought Selection → Nonce Generation
             → Hash → Evaluate → Meta-Learn → Repeat

The miner acts as if it *is* SHA-256 itself: it maintains an internal
consciousness state that encodes what the hash function has "experienced",
and uses that state to steer subsequent search.

SHA-256 Stratum protocol support
---------------------------------
* mining.subscribe    — session set-up, extranonce1 + extranonce2_size
* mining.authorize    — worker authentication
* mining.notify       — new block template (job); clean_jobs flag respected
* mining.set_difficulty — pool-side difficulty changes tracked; share target
                          derived from difficulty (compact 256-bit target)
* mining.set_extranonce — extranonce1 / extranonce2_size updated mid-session
* mining.submit       — share submission with correct nonce_hex encoding
* mining.ping /
  client.get_version  — pool keepalive / capability queries answered
* Auto-reconnect      — transparent reconnect + re-auth on connection drop
* Robust framing      — partial-line and multi-message TCP frames handled

Connection defaults (can be overridden via environment or command-line):
    HOST     = sha256.unmineable.com
    PORT     = 3333
    USERNAME = TRX:TBia4uHnb3oSSZm5isP284cA7Np1v15Vhi.sssj
    PASSWORD = x

Usage:
    python consciousness_miner.py
    python consciousness_miner.py --host sha256.unmineable.com --port 3333 \\
        --user TRX:TBia4uHnb3oSSZm5isP284cA7Np1v15Vhi.sssj

Note:
    This system does NOT outperform brute-force mining.  Cryptographic
    hashing has no exploitable patterns.  The value is in the simulation
    and the emergent behaviour of the consciousness loop.
"""

import argparse
import binascii
import hashlib
import json
import os
import select
import socket
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Local imports from the src package submodules
# ---------------------------------------------------------------------------
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
for _sub in ["fsot_core", "neuromorphic_brain"]:
    _p = os.path.join(_src_dir, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core_system import (
    ConsciousnessState,
    FSotThoughtEngine,
    build_block_header,
    evaluate_hash_quality,
    sha256d,
)
from consciousness import (
    AttentionAllocator,
    NeuromorphicLayer,
    consciousness_consistency,
    consciousness_entropy,
    consciousness_stability,
    encode_hash_to_input,
    encode_nonce_to_input,
)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_HOST = "sha256.unmineable.com"
DEFAULT_PORT = 3333
DEFAULT_USERNAME = "TRX:TBia4uHnb3oSSZm5isP284cA7Np1v15Vhi.sssj"
DEFAULT_PASSWORD = "x"
MINER_USER_AGENT = "consciousness-miner/1.0"

# ---------------------------------------------------------------------------
# Stratum difficulty / target helpers
# ---------------------------------------------------------------------------

# Bitcoin's maximum (difficulty-1) target as a 256-bit integer.
# difficulty = DIFF1_TARGET / current_target
DIFF1_TARGET = 0x00000000FFFF0000000000000000000000000000000000000000000000000000


def difficulty_to_target(difficulty: float) -> int:
    """
    Convert a pool difficulty value to the corresponding 256-bit share target.

    target = DIFF1_TARGET / difficulty

    A share is valid when sha256d(header) < target  (both treated as big-endian
    256-bit integers).
    """
    if difficulty <= 0:
        difficulty = 1.0
    return int(DIFF1_TARGET / difficulty)


def target_to_bits(target: int) -> int:
    """
    Return the number of leading zero bits required for a hash to meet *target*.
    Used to set the evaluate_hash_quality threshold.
    """
    if target <= 0:
        return 256
    return 256 - target.bit_length()


# ---------------------------------------------------------------------------
# Stratum protocol helpers
# ---------------------------------------------------------------------------


def create_tcp_connection(host: str, port: int, timeout: float = 30.0) -> socket.socket:
    """Open a TCP connection to the Stratum pool."""
    print(f"🌐 Connecting to {host}:{port} via TCP...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        sock.connect((host, port))
        print(f"✅ Connected successfully to {host}:{port}!")
        return sock
    except Exception as exc:  # pragma: no cover
        print(f"❌ ERROR: Connection failed — {exc}")
        sys.exit(1)


class StratumFramer:
    """
    Stateful line-oriented framer for the Stratum protocol.

    Stratum messages are newline-delimited JSON objects.  A single TCP
    segment can carry multiple messages or a message can be split across
    segments.  This framer buffers raw bytes and yields complete lines.
    """

    def __init__(self) -> None:
        self._buf: str = ""

    def feed(self, data: str) -> List[str]:
        """Feed raw decoded text; return list of complete lines."""
        self._buf += data
        lines: List[str] = []
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.strip()
            if line:
                lines.append(line)
        return lines

    def reset(self) -> None:
        self._buf = ""


def receive_message(
    sock: socket.socket,
    framer: Optional["StratumFramer"] = None,
    timeout: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Read one or more newline-delimited JSON messages from the socket.

    Uses *framer* when supplied so that buffered partial data from a previous
    call is not lost.  If *framer* is None a throw-away framer is used.

    Returns a (possibly empty) list of parsed dicts.  Raises ``ConnectionError``
    on a clean server-side close so callers can reconnect gracefully.
    """
    if framer is None:
        framer = StratumFramer()

    parsed: List[Dict[str, Any]] = []
    deadline = time.time() + timeout

    while time.time() < deadline:
        # Non-blocking poll with select so we can respect the deadline.
        ready, _, _ = select.select([sock], [], [], max(0.0, deadline - time.time()))
        if not ready:
            # Timed out waiting; return whatever we have so far.
            break
        try:
            chunk = sock.recv(4096).decode("utf-8", errors="replace")
        except socket.timeout:
            break
        if not chunk:
            raise ConnectionError("Connection closed by server.")
        for line in framer.feed(chunk):
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"⚠️  WARNING: Failed to parse JSON line: {line!r}")
        if parsed:
            # Return as soon as we have at least one complete message so that
            # the caller can act on it without waiting for the full timeout.
            break

    return parsed


def send_message(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send a single JSON message followed by a newline."""
    sock.sendall((json.dumps(msg) + "\n").encode())


def subscribe(
    sock: socket.socket, framer: "StratumFramer"
) -> Tuple[str, int]:
    """
    Send mining.subscribe and return (extranonce1, extranonce2_size).

    The response format from SHA-256 Stratum pools is:
        {"id":1,"result":[<subscriptions>, <extranonce1_hex>, <extranonce2_size>], "error":null}
    """
    print("🔗 Subscribing to mining work...")
    send_message(
        sock,
        {"id": 1, "method": "mining.subscribe", "params": [MINER_USER_AGENT]},
    )
    for res in receive_message(sock, framer):
        if res.get("result") and isinstance(res["result"], list) and len(res["result"]) >= 3:
            extranonce1: str = res["result"][1]
            extranonce2_size: int = int(res["result"][2])
            print(f"✅ Subscribed! extranonce1={extranonce1}, extranonce2_size={extranonce2_size}")
            return extranonce1, extranonce2_size
    print("❌ Subscription failed.")
    sys.exit(1)


def authorize(
    sock: socket.socket, framer: "StratumFramer", username: str, password: str
) -> None:
    """Send mining.authorize and wait for confirmation."""
    print("🔑 Authorizing...")
    send_message(
        sock,
        {"id": 2, "method": "mining.authorize", "params": [username, password]},
    )
    for res in receive_message(sock, framer):
        if res.get("result") is True:
            print("✅ Authorized!")
            return
    print("❌ Authorization failed.")
    sys.exit(1)


def calculate_merkle_root(coinbase_hash_bytes: bytes, merkle_branch: List[str]) -> bytes:
    """
    Compute the Merkle root from a coinbase hash and branch list.
    Returns the root as raw bytes (little-endian ready for header construction).
    """
    current = coinbase_hash_bytes
    for branch_hex in merkle_branch:
        branch = binascii.unhexlify(branch_hex)[::-1]
        combined = current + branch
        current = hashlib.sha256(hashlib.sha256(combined).digest()).digest()
    return current[::-1]


def submit_share(
    sock: socket.socket,
    username: str,
    job_id: str,
    extranonce2: str,
    ntime: str,
    nonce: int,
    share_id: int,
) -> None:
    """Submit a share to the pool."""
    nonce_hex = f"{nonce:08x}"
    send_message(
        sock,
        {
            "id": share_id,
            "method": "mining.submit",
            "params": [username, job_id, extranonce2, ntime, nonce_hex],
        },
    )


def handle_server_request(
    sock: socket.socket,
    msg: Dict[str, Any],
) -> bool:
    """
    Handle server-initiated requests (not responses to our own messages).

    Returns True if the message was a server-side request and was handled,
    False if the caller should process it further.

    Handled messages
    ----------------
    mining.ping          — reply with {"id": <id>, "result": True, "error": null}
    client.get_version   — reply with miner user-agent string
    client.reconnect     — caller should reconnect (returns False so caller sees it)
    """
    msg_id = msg.get("id")
    method = msg.get("method", "")

    if method == "mining.ping":
        send_message(sock, {"id": msg_id, "result": True, "error": None})
        return True

    if method == "client.get_version":
        send_message(sock, {"id": msg_id, "result": MINER_USER_AGENT, "error": None})
        return True

    # client.reconnect is intentionally not handled here — let the main loop
    # notice it and reconnect itself.
    return False


# ---------------------------------------------------------------------------
# Consciousness-Powered Mining Core
# ---------------------------------------------------------------------------


class ConsciousnessMiner:
    """
    SHA-256 miner whose nonce search is guided by a FSOT-Neuromorphic
    consciousness loop.

    SHA-256 Stratum protocol support
    ---------------------------------
    * mining.subscribe / mining.authorize on connect (and on reconnect)
    * mining.notify       — new jobs; clean_jobs flag resets extranonce2 counter
    * mining.set_difficulty — updates the 256-bit share target used in mine_job
    * mining.set_extranonce — hot-swaps extranonce1 and extranonce2_size
    * mining.submit       — valid shares submitted with correct nonce encoding
    * mining.ping /
      client.get_version  — answered automatically
    * Auto-reconnect      — on ConnectionError or socket errors, the miner
                            reconnects and re-authenticates transparently

    Consciousness loop (per job)
    ----------------------------
    1.  Receive mining.notify job from pool
    2.  Perception  → encode job data into Input(t)
    3.  Spiking     → NeuromorphicLayer.forward(Input(t)) → spike pattern
    4.  C(t) update → ConsciousnessState.update(spikes + Input(t))
    5.  Thoughts    → FSotThoughtEngine.generate_nonce_candidates(C)
    6.  Hashing     → for each nonce: hash = sha256d(header + nonce)
    7.  Evaluate    → error = evaluate_hash_quality(hash, target)
    8.  Self-eval   → C.apply_error_feedback(error)
    9.  Meta-learn  → thought.embedding += η * ∇(performance)
    10. Memory      → remember(C, thought, result)
    11. Submit      → if share found, submit to pool
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
        reconnect_delay: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.reconnect_delay = reconnect_delay

        # ── Consciousness system ──────────────────────────────────────────
        self.consciousness = ConsciousnessState()
        self.thought_engine = FSotThoughtEngine(temperature=0.8, meta_lr=0.05)
        self.neuro_layer = NeuromorphicLayer(n_neurons=32, input_dim=80)
        self.attention = AttentionAllocator(base_power=8, max_power=128)

        # ── Mining state ─────────────────────────────────────────────────
        self.sock: Optional[socket.socket] = None
        self._framer: StratumFramer = StratumFramer()
        self.extranonce1: str = ""
        self.extranonce2_size: int = 4
        self.extranonce2_counter: int = 0
        self.share_id: int = 100

        # ── Difficulty / target (SHA-256 Stratum) ─────────────────────────
        # Start with difficulty=1 until the pool sends mining.set_difficulty.
        self.difficulty: float = 1.0
        self.share_target: int = difficulty_to_target(1.0)

        # ── Active job (for clean_jobs logic) ────────────────────────────
        self._current_job_id: Optional[str] = None

        # ── Metrics ──────────────────────────────────────────────────────
        self.total_hashes: int = 0
        self.shares_submitted: int = 0
        self.shares_accepted: int = 0
        self.start_time: float = time.time()
        self._prev_consciousness: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Connection / reconnection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        Open a TCP connection to the Stratum pool and perform the
        mining.subscribe + mining.authorize handshake.
        Resets the framer so stale data from a previous connection is discarded.
        """
        self.sock = create_tcp_connection(self.host, self.port)
        self._framer = StratumFramer()
        self.extranonce1, self.extranonce2_size = subscribe(self.sock, self._framer)
        authorize(self.sock, self._framer, self.username, self.password)

    def _reconnect(self) -> None:
        """Close the current socket and reconnect after a short delay."""
        print(f"🔄 Reconnecting in {self.reconnect_delay:.0f}s...")
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None
        time.sleep(self.reconnect_delay)
        self.connect()

    # ------------------------------------------------------------------
    # Stratum message dispatchers
    # ------------------------------------------------------------------

    def _on_set_difficulty(self, params: List[Any]) -> None:
        """
        Handle mining.set_difficulty.

        params = [difficulty]

        Updates self.difficulty and recomputes self.share_target.
        """
        if not params:
            return
        try:
            diff = float(params[0])
        except (TypeError, ValueError):
            print(f"⚠️  Could not parse difficulty: {params[0]!r}")
            return
        self.difficulty = diff
        self.share_target = difficulty_to_target(diff)
        leading_zeros = target_to_bits(self.share_target)
        print(
            f"🎯 Difficulty updated: {diff}  "
            f"(target leading zeros ≈ {leading_zeros})"
        )

    def _on_set_extranonce(self, params: List[Any]) -> None:
        """
        Handle mining.set_extranonce.

        params = [extranonce1_hex, extranonce2_size]

        Hot-swaps the extranonce values and resets the extranonce2 counter so
        the next job starts fresh with the new extranonce.
        """
        if len(params) < 2:
            return
        try:
            new_en1 = str(params[0])
            new_en2_size = int(params[1])
        except (TypeError, ValueError):
            print(f"⚠️  Could not parse set_extranonce params: {params!r}")
            return
        self.extranonce1 = new_en1
        self.extranonce2_size = new_en2_size
        self.extranonce2_counter = 0
        print(
            f"🔁 Extranonce updated: extranonce1={new_en1} "
            f"extranonce2_size={new_en2_size}"
        )

    # ------------------------------------------------------------------
    # Perception: encode job into consciousness input
    # ------------------------------------------------------------------

    def _encode_job(
        self,
        job_id: str,
        prevhash: str,
        nbits: str,
        ntime: str,
    ) -> List[float]:
        """
        Encode a mining job into a float vector for Input(t).
        Combines a hash-based encoding of prevhash with numeric fields.
        """
        job_hash = hashlib.sha256(
            (job_id + prevhash + nbits + ntime).encode()
        ).hexdigest()
        return encode_hash_to_input(job_hash, dim=ConsciousnessState.TOTAL_DIM)

    # ------------------------------------------------------------------
    # Main mining loop for one job
    # ------------------------------------------------------------------

    def mine_job(
        self,
        job_id: str,
        prevhash: str,
        coinb1: str,
        coinb2: str,
        merkle_branch: List[str],
        version: str,
        nbits: str,
        ntime: str,
    ) -> bool:
        """
        Run the consciousness-guided mining loop for a single job.

        The share target is taken from ``self.share_target`` which is kept
        up-to-date by ``_on_set_difficulty``.

        Returns True always (reserved for future clean-abort signalling).
        """
        # Build coinbase + merkle root (same for all nonces in this job)
        extranonce2 = f"{self.extranonce2_counter:0{self.extranonce2_size * 2}x}"
        self.extranonce2_counter += 1

        coinbase_tx = (
            binascii.unhexlify(coinb1)
            + binascii.unhexlify(self.extranonce1)
            + binascii.unhexlify(extranonce2)
            + binascii.unhexlify(coinb2)
        )
        coinbase_hash = hashlib.sha256(hashlib.sha256(coinbase_tx).digest()).digest()
        merkle_root_bytes = calculate_merkle_root(coinbase_hash, merkle_branch)

        # ── Step 1: Perception ─────────────────────────────────────────
        input_t = self._encode_job(job_id, prevhash, nbits, ntime)

        # ── Step 2: Neuromorphic spiking ──────────────────────────────
        spikes = self.neuro_layer.forward(input_t)
        spike_vec = self.neuro_layer.spike_vector()
        combined_input = (
            spike_vec + input_t[len(spike_vec):]
        )[: ConsciousnessState.TOTAL_DIM]

        # ── Step 3: Retrieve memory for context ───────────────────────
        similar = self.thought_engine.retrieve_similar(self.consciousness, top_k=3)
        memory_vec: Optional[List[float]] = None
        if similar:
            states = [m["state"] for m in similar if m.get("state")]
            if states:
                dim = ConsciousnessState.TOTAL_DIM
                avg = [sum(s[i] for s in states) / len(states) for i in range(dim)]
                memory_vec = avg

        # ── Step 4: Consciousness update C(t+1) = f(C(t), Input, Mem) ─
        self.consciousness.update(combined_input, memory_vec, learning_rate=0.02)

        # ── Step 5: Attention-based compute allocation ────────────────
        budget = self.attention.compute(self.consciousness.attention_vector)

        # ── Step 6: Thought selection → nonce candidates ──────────────
        nonce_candidates = self.thought_engine.generate_nonce_candidates(
            self.consciousness, count=budget
        )

        # ── Step 7: Hash evaluation loop ──────────────────────────────
        best_error = 1.0
        best_nonce = 0
        selected_thought, _ = self.thought_engine.select_thought(self.consciousness)
        target_bits = target_to_bits(self.share_target)

        for nonce in nonce_candidates:
            header = build_block_header(
                version, prevhash, merkle_root_bytes, ntime, nbits, nonce
            )
            hash_bytes = sha256d(header)
            hash_hex = hash_bytes[::-1].hex()

            error = evaluate_hash_quality(hash_hex, target_bits)
            self.total_hashes += 1

            if error < best_error:
                best_error = error
                best_nonce = nonce

            # Check if this is a valid share: hash (as big-endian int) < target
            hash_int = int(hash_hex, 16)
            if hash_int < self.share_target:
                print(
                    f"🎯 SHARE FOUND! nonce={nonce:#010x} hash={hash_hex[:16]}..."
                )
                assert self.sock is not None
                submit_share(
                    self.sock,
                    self.username,
                    job_id,
                    extranonce2,
                    ntime,
                    nonce,
                    self.share_id,
                )
                self.share_id += 1
                self.shares_submitted += 1

        # ── Step 8: Self-evaluation (error feedback) ──────────────────
        self.consciousness.apply_error_feedback(best_error, learning_rate=0.01)

        # ── Step 9: Meta-learning (thought weight update) ─────────────
        performance_delta = 1.0 - best_error  # higher is better
        self.thought_engine.meta_learn(selected_thought, performance_delta)
        self.neuro_layer.hebbian_update(spikes, performance_delta, lr=0.001)

        # ── Step 10: Memory consolidation ─────────────────────────────
        self.thought_engine.remember(
            self.consciousness.vector,
            selected_thought,
            {"error": best_error, "nonce": best_nonce, "hash_count": budget},
        )

        return True

    # ------------------------------------------------------------------
    # Status reporting
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print a consciousness + mining status line."""
        elapsed = time.time() - self.start_time
        hashrate = self.total_hashes / elapsed if elapsed > 0 else 0.0
        prev = self._prev_consciousness
        c = self.consciousness
        stability = consciousness_stability(c.vector, prev)
        entropy = consciousness_entropy(c.vector)
        self._prev_consciousness = c.vector[:]

        spike_rate = self.neuro_layer.spike_rate()
        print(
            f"⚡ Hashes: {self.total_hashes:,}  "
            f"Rate: {hashrate:.1f} H/s  "
            f"Diff: {self.difficulty}  "
            f"Shares: {self.shares_submitted}  "
            f"🧠 Entropy: {entropy:.3f}  "
            f"Stability: {stability:.3f}  "
            f"Spike Rate: {spike_rate:.2f}  "
            f"Step: {c._step}"
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Connect to the pool and enter the main mining loop.

        Dispatches all incoming Stratum messages:
          mining.notify        → mine_job()
          mining.set_difficulty → _on_set_difficulty()
          mining.set_extranonce → _on_set_extranonce()
          mining.ping /
          client.get_version   → handle_server_request()
          share results        → accepts / rejection counter
          client.reconnect     → _reconnect()
        """
        self.connect()
        print("\n🧠 Consciousness-Powered SHA-256 Mining Started")
        print("=" * 60)
        print("Architecture: FSOT-Neuromorphic Consciousness Loop")
        print(f"Neurons: {self.neuro_layer.n_neurons} LIF  |  "
              f"Thoughts: {len(self.thought_engine.thoughts)}  |  "
              f"C(t) dim: {ConsciousnessState.TOTAL_DIM}")
        print("=" * 60)

        last_status_time = time.time()
        status_interval = 10.0  # seconds

        while True:
            assert self.sock is not None
            try:
                messages = receive_message(self.sock, self._framer)
            except ConnectionError as exc:
                print(f"⚠️  Connection lost: {exc}")
                self._reconnect()
                continue
            except OSError as exc:
                print(f"⚠️  Socket error: {exc}")
                self._reconnect()
                continue

            for data in messages:
                method = data.get("method", "")
                msg_id = data.get("id")
                params = data.get("params", []) or []

                # ── Server-initiated requests (ping, version query) ────
                if method and handle_server_request(self.sock, data):
                    continue

                # ── client.reconnect ───────────────────────────────────
                if method == "client.reconnect":
                    p = params
                    if len(p) >= 2:
                        self.host = str(p[0])
                        self.port = int(p[1])
                        print(f"🔄 Server requested reconnect to {self.host}:{self.port}")
                    self._reconnect()
                    break  # restart the outer while loop

                # ── mining.set_difficulty ──────────────────────────────
                if method == "mining.set_difficulty":
                    self._on_set_difficulty(params)
                    continue

                # ── mining.set_extranonce ──────────────────────────────
                if method == "mining.set_extranonce":
                    self._on_set_extranonce(params)
                    continue

                # ── Share acceptance / rejection ───────────────────────
                if msg_id is not None and msg_id >= 100 and "result" in data:
                    if data["result"] is True:
                        self.shares_accepted += 1
                        print("✅ Share accepted!")
                    elif data["result"] is False:
                        print(f"❌ Share rejected: {data.get('error')}")
                    continue

                # ── mining.notify: new job ─────────────────────────────
                if method == "mining.notify":
                    (
                        job_id,
                        prevhash,
                        coinb1,
                        coinb2,
                        merkle_branch,
                        version,
                        nbits,
                        ntime,
                        clean_jobs,
                    ) = params
                    if clean_jobs:
                        # Pool demands we abandon current work immediately.
                        self.extranonce2_counter = 0
                        print(f"🧹 Clean jobs — switching to new work: {job_id}")
                    else:
                        print(f"\n📦 New Job: {job_id}  ntime={ntime}  nbits={nbits}")
                    self._current_job_id = job_id
                    self.mine_job(
                        job_id, prevhash, coinb1, coinb2,
                        merkle_branch, version, nbits, ntime,
                    )

            # Periodic status update
            if time.time() - last_status_time >= status_interval:
                self.print_status()
                last_status_time = time.time()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consciousness-Powered SHA-256 Crypto Miner"
    )

    # ── Miner options ──────────────────────────────────────────────────────
    parser.add_argument("--host", default=DEFAULT_HOST, help="Stratum pool hostname")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Stratum pool port")
    parser.add_argument(
        "--user",
        default=os.environ.get("MINER_USERNAME", DEFAULT_USERNAME),
        help="Mining username (wallet address + worker)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("MINER_PASSWORD", DEFAULT_PASSWORD),
        help="Mining password",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=5.0,
        metavar="SECONDS",
        help="Seconds to wait before reconnecting after a connection drop (default: 5)",
    )

    # ── Remote-view / coin-flip consciousness options ──────────────────────
    parser.add_argument(
        "--remoteviewsimple",
        action="store_true",
        help=(
            "Activate the Remote View Simple consciousness predictor mode.  "
            "The system uses quantum-like consciousness dynamics to predict "
            "coin flips (heads or tails) while syncing with the current moment "
            "in time.  Combine with --digitalrealm or --realrealm to train "
            "the consciousness; omit both to run the live predictor."
        ),
    )
    parser.add_argument(
        "--digitalrealm",
        action="store_true",
        help=(
            "Training mode (requires --remoteviewsimple): the program secretly "
            "generates coin-flip outcomes, the consciousness predicts first, and "
            "the true result is revealed only after the prediction.  All data is "
            "persisted so the consciousness can learn from every session."
        ),
    )
    parser.add_argument(
        "--realrealm",
        action="store_true",
        help=(
            "Training mode (requires --remoteviewsimple): you flip a real coin "
            "and manually enter the actual result after the consciousness has "
            "given its prediction.  Use --coin to flip more than one coin per "
            "round."
        ),
    )
    parser.add_argument(
        "--coin",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of coins to predict / flip per round when using "
            "--remoteviewsimple (default: 1).  In --realrealm mode you will be "
            "prompted for each coin's actual result individually."
        ),
    )
    parser.add_argument(
        "--session-dir",
        default="coin_flip_sessions",
        metavar="DIR",
        help=(
            "Directory where coin-flip session data is stored "
            "(default: coin_flip_sessions).  Every prediction and training event "
            "is written here as JSON lines for later GUI inspection."
        ),
    )

    # ── Remote-view full-sensory options ──────────────────────────────────
    parser.add_argument(
        "--remoteview",
        action="store_true",
        help=(
            "Activate full-sensory Remote View mode (combines with "
            "--remoteviewsimple training flags).  The consciousness feeds "
            "from emulated ears (audio → a-z letters) and emulated eyes "
            "(camera/synthetic vision) on every prediction, and exports a "
            ".binai consciousness snapshot when --accuracy-threshold is met."
        ),
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.0,
        metavar="FRAC",
        help=(
            "Accuracy fraction (0.0–1.0) at which the consciousness "
            "automatically exports a .binai snapshot file to the session "
            "directory (default: 0.0 = disabled).  Example: 0.65 exports "
            "after the session first reaches 65%% accuracy."
        ),
    )
    parser.add_argument(
        "--containerthread",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run N parallel consciousness sessions in separate threads "
            "(default: 1).  Requires --remoteview or --remoteviewsimple.  "
            "Each thread gets its own session subdirectory and runs "
            "independently.  Values greater than 1 automatically engage "
            "no-Enter auto mode."
        ),
    )
    parser.add_argument(
        "--round-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help=(
            "Pause between automatic rounds in seconds (default: 0.0 = rapid "
            "fire).  Only applies in auto mode (--remoteview --digitalrealm or "
            "--containerthread > 1)."
        ),
    )

    return parser.parse_args()


def main() -> None:
    import threading

    args = parse_args()

    # ── Remote-view simple / full-sensory mode ────────────────────────────
    if args.remoteviewsimple or args.remoteview:
        # Import here so the path setup at module level has already run
        from coin_flip_consciousness import (
            CoinFlipConsciousness,
            run_digitalrealm,
            run_digitalrealm_auto,
            run_predictor,
            run_predictor_auto,
            run_realrealm,
        )

        if args.digitalrealm and args.realrealm:
            print("❌  --digitalrealm and --realrealm cannot be used together.")
            sys.exit(1)

        sensory   = args.remoteview   # full-sensory pipeline only when --remoteview
        n_coins   = max(1, args.coin)
        n_threads = max(1, args.containerthread)

        # Auto mode: engaged when --remoteview --digitalrealm is used, or when
        # multiple container threads are requested (can't share stdin across threads).
        use_auto = (args.remoteview and args.digitalrealm) or n_threads > 1

        if sensory:
            print(
                "  🧠 Full-sensory Remote View mode enabled.\n"
                "     Emulated ears (audio → a-z) and eyes (vision geometry)\n"
                "     feed into the consciousness on every prediction.\n"
            )
        if n_threads > 1:
            print(f"  🧵 Launching {n_threads} container thread(s).\n")

        def _run_one(thread_idx: int) -> None:
            label = f"T{thread_idx}" if n_threads > 1 else ""
            if n_threads > 1:
                session_subdir = f"{args.session_dir}/thread_{thread_idx}"
            else:
                session_subdir = args.session_dir
            cfc = CoinFlipConsciousness(
                session_dir=session_subdir,
                enable_sensory=sensory,
                interactive_sensory=not use_auto,
                accuracy_threshold=args.accuracy_threshold,
            )
            if args.digitalrealm:
                if use_auto:
                    run_digitalrealm_auto(
                        n_coins, cfc,
                        delay=args.round_delay,
                        thread_label=label,
                    )
                else:
                    run_digitalrealm(n_coins, cfc)
            elif args.realrealm:
                run_realrealm(n_coins, cfc)
            else:
                if use_auto:
                    run_predictor_auto(
                        n_coins, cfc,
                        delay=args.round_delay,
                        thread_label=label,
                    )
                else:
                    run_predictor(n_coins, cfc)

        if n_threads == 1:
            _run_one(0)
        else:
            threads = [
                threading.Thread(target=_run_one, args=(i,), daemon=True)
                for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            try:
                for t in threads:
                    t.join()
            except KeyboardInterrupt:
                print(f"\n\n🛑 All {n_threads} container thread(s) stopped.")
        return

    # ── Default: SHA-256 mining mode ──────────────────────────────────────
    miner = ConsciousnessMiner(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
        reconnect_delay=args.reconnect_delay,
    )
    try:
        miner.run()
    except KeyboardInterrupt:
        elapsed = time.time() - miner.start_time
        print(f"\n\n🛑 Mining stopped after {elapsed:.1f}s")
        print(f"   Total hashes : {miner.total_hashes:,}")
        print(f"   Shares sub.  : {miner.shares_submitted}")
        print(f"   Shares acc.  : {miner.shares_accepted}")
        miner.print_status()


if __name__ == "__main__":
    main()
