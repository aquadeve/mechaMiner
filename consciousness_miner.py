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

# ---------------------------------------------------------------------------
# Stratum protocol helpers
# ---------------------------------------------------------------------------


def create_tcp_connection(host: str, port: int) -> socket.socket:
    """Open a TCP connection to the Stratum pool."""
    print(f"🌐 Connecting to {host}:{port} via TCP...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30)
        sock.connect((host, port))
        print(f"✅ Connected successfully to {host}:{port}!")
        return sock
    except Exception as exc:  # pragma: no cover
        print(f"❌ ERROR: Connection failed — {exc}")
        sys.exit(1)


def receive_message(sock: socket.socket) -> List[Dict[str, Any]]:
    """
    Read one or more newline-delimited JSON messages from the socket.
    Returns a list of parsed dicts.
    """
    response = ""
    try:
        while True:
            part = sock.recv(4096).decode("utf-8")
            if not part:
                print("❌ ERROR: Connection closed by server.")
                sys.exit(1)
            response += part
            if "\n" in response:
                break
    except socket.timeout:
        print("❌ ERROR: Socket timeout.")
        sys.exit(1)

    parsed: List[Dict[str, Any]] = []
    for line in response.strip().split("\n"):
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"⚠️  WARNING: Failed to parse JSON line: {line!r}")
    return parsed


def send_message(sock: socket.socket, msg: Dict[str, Any]) -> None:
    """Send a single JSON message followed by a newline."""
    sock.sendall((json.dumps(msg) + "\n").encode())


def subscribe(sock: socket.socket) -> Tuple[str, int]:
    """
    Send mining.subscribe and return (extranonce1, extranonce2_size).
    """
    print("🔗 Subscribing to mining work...")
    send_message(
        sock,
        {"id": 1, "method": "mining.subscribe", "params": ["consciousness-miner/1.0"]},
    )
    for res in receive_message(sock):
        if res.get("result") and isinstance(res["result"], list) and len(res["result"]) >= 3:
            extranonce1: str = res["result"][1]
            extranonce2_size: int = int(res["result"][2])
            print(f"✅ Subscribed! extranonce1={extranonce1}, extranonce2_size={extranonce2_size}")
            return extranonce1, extranonce2_size
    print("❌ Subscription failed.")
    sys.exit(1)


def authorize(sock: socket.socket, username: str, password: str) -> None:
    """Send mining.authorize."""
    print("🔑 Authorizing...")
    send_message(
        sock,
        {"id": 2, "method": "mining.authorize", "params": [username, password]},
    )
    for res in receive_message(sock):
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


# ---------------------------------------------------------------------------
# Consciousness-Powered Mining Core
# ---------------------------------------------------------------------------


class ConsciousnessMiner:
    """
    SHA-256 miner whose nonce search is guided by a FSOT-Neuromorphic
    consciousness loop.

    Core loop
    ---------
    1. Receive mining.notify job from pool
    2. Perception  → encode job data into Input(t)
    3. Spiking     → NeuromorphicLayer.forward(Input(t)) → spike pattern
    4. C(t) update → ConsciousnessState.update(spikes + Input(t))
    5. Thoughts    → FSotThoughtEngine.generate_nonce_candidates(C)
    6. Hashing     → for each nonce: hash = sha256d(header + nonce)
    7. Evaluate    → error = evaluate_hash_quality(hash)
    8. Self-eval   → C.apply_error_feedback(error)
    9. Meta-learn  → thought.embedding += η * ∇(performance)
    10. Memory     → remember(C, thought, result)
    11. Submit     → if share found, submit to pool
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        username: str = DEFAULT_USERNAME,
        password: str = DEFAULT_PASSWORD,
    ) -> None:
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        # ── Consciousness system ──────────────────────────────────────────
        self.consciousness = ConsciousnessState()
        self.thought_engine = FSotThoughtEngine(temperature=0.8, meta_lr=0.05)
        self.neuro_layer = NeuromorphicLayer(n_neurons=32, input_dim=80)
        self.attention = AttentionAllocator(base_power=8, max_power=128)

        # ── Mining state ─────────────────────────────────────────────────
        self.sock: Optional[socket.socket] = None
        self.extranonce1: str = ""
        self.extranonce2_size: int = 4
        self.extranonce2_counter: int = 0
        self.share_id: int = 100

        # ── Metrics ──────────────────────────────────────────────────────
        self.total_hashes: int = 0
        self.shares_submitted: int = 0
        self.shares_accepted: int = 0
        self.start_time: float = time.time()
        self._prev_consciousness: Optional[List[float]] = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the Stratum pool and perform subscribe + authorize."""
        self.sock = create_tcp_connection(self.host, self.port)
        self.extranonce1, self.extranonce2_size = subscribe(self.sock)
        authorize(self.sock, self.username, self.password)

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
        # Hash the job fields to get a compact representation
        job_hash = hashlib.sha256(
            (job_id + prevhash + nbits + ntime).encode()
        ).hexdigest()
        return encode_hash_to_input(job_hash, dim=ConsciousnessState.TOTAL_DIM)

    # ------------------------------------------------------------------
    # Main mining loop
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
        target_bits: int = 32,
    ) -> bool:
        """
        Run the consciousness-guided mining loop for a single job.
        Returns True if the job is still active (no new job received),
        False if a new job notification interrupted.
        """
        # Build coinbase + merkle root (same for all nonces)
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
        spike_rate = self.neuro_layer.spike_rate()
        spike_vec = self.neuro_layer.spike_vector()
        # Pad spike_vec to match ConsciousnessState.TOTAL_DIM
        combined_input = (
            spike_vec + input_t[len(spike_vec):]
        )[: ConsciousnessState.TOTAL_DIM]

        # ── Step 3: Retrieve memory for context ───────────────────────
        similar = self.thought_engine.retrieve_similar(self.consciousness, top_k=3)
        memory_vec: Optional[List[float]] = None
        if similar:
            # Average the state vectors of similar memories
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

            # Check if this is a valid share (leading zeros ≥ target_bits)
            hash_int = int(hash_hex, 16)
            if hash_int < (2 ** (256 - target_bits)):
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
        Handles incoming mining.notify messages and dispatches to mine_job().
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
            messages = receive_message(self.sock)
            for data in messages:
                # Handle share acceptance/rejection
                if data.get("id", 0) >= 100 and "result" in data:
                    if data["result"] is True:
                        self.shares_accepted += 1
                        print("✅ Share accepted!")
                    elif data["result"] is False:
                        print(f"❌ Share rejected: {data.get('error')}")

                # Handle new job notification
                if data.get("method") == "mining.notify":
                    params = data["params"]
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
                    print(f"\n📦 New Job: {job_id}  ntime={ntime}  nbits={nbits}")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    miner = ConsciousnessMiner(
        host=args.host,
        port=args.port,
        username=args.user,
        password=args.password,
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
