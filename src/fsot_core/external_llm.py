#!/usr/bin/env python3
"""
Lightweight external LLM integration for the FSOT/consciousness stack.

The adapter can:
  * call an external chat-completions style HTTP endpoint when configured
  * fall back to a tiny trainable local language model when no endpoint exists
  * reuse the existing consciousness-backed memory/session store as context

The local model is deliberately simple so it can be trained in local shells and
GitHub Actions without heavyweight dependencies.
"""

from __future__ import annotations

import json
import math
import os
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from neuromorphic_brain.consciousness import ConsciousnessBackedLLM
except ImportError:  # pragma: no cover - used by direct-path tests
    from consciousness import ConsciousnessBackedLLM


DEFAULT_TRAINING_SAMPLES: List[Dict[str, str]] = [
    {
        "prompt": "summarize the current mining session",
        "response": "The mining session is stable and can be summarized through recent consciousness traces.",
    },
    {
        "prompt": "remember the last performance benchmark",
        "response": "I remember the last benchmark and can surface the strongest signals from stored memory.",
    },
    {
        "prompt": "explain the consciousness memory system",
        "response": "The consciousness memory system stores thoughts, emotions, and episodic traces in local files for reuse.",
    },
    {
        "prompt": "help me plan a training run",
        "response": "Use the lightweight local trainer to fit a checkpoint, then load it for local or CI inference.",
    },
    {
        "prompt": "how do i train this model in github actions",
        "response": "Run the lightweight training script in GitHub Actions to build a JSON checkpoint artifact.",
    },
]


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _vectorize(tokens: Iterable[str]) -> Dict[str, float]:
    counts: Dict[str, float] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0.0) + 1.0
    return counts


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(key, 0.0) for key, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def load_training_samples(dataset_path: Optional[str] = None) -> List[Dict[str, str]]:
    if not dataset_path:
        return list(DEFAULT_TRAINING_SAMPLES)

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("JSON dataset must be a list of {prompt, response} objects")
        return [dict(item) for item in payload]

    samples: List[Dict[str, str]] = []
    for line in text.splitlines():
        if line.strip():
            samples.append(dict(json.loads(line)))
    return samples


class TinyLocalLanguageModel:
    """
    Tiny trainable text model based on token overlap retrieval.

    It stores prompt/response exemplars and chooses the best response by cosine
    similarity over bag-of-words vectors. This is intentionally lightweight.
    """

    def __init__(
        self,
        model_name: str = "tiny-consciousness",
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.examples: List[Dict[str, Any]] = []
        self.checkpoint_path = checkpoint_path
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def train(self, samples: List[Dict[str, str]]) -> Dict[str, Any]:
        prepared: List[Dict[str, Any]] = []
        vocabulary = set()
        for sample in samples:
            prompt = str(sample.get("prompt", "")).strip()
            response = str(sample.get("response", "")).strip()
            if not prompt or not response:
                continue
            prompt_tokens = _tokenize(prompt)
            response_tokens = _tokenize(response)
            vocabulary.update(prompt_tokens)
            vocabulary.update(response_tokens)
            prepared.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "prompt_vector": _vectorize(prompt_tokens),
                    "response_vector": _vectorize(response_tokens),
                }
            )

        self.examples = prepared
        return {
            "model_name": self.model_name,
            "sample_count": len(self.examples),
            "vocabulary_size": len(vocabulary),
            "trained_at": datetime.now().astimezone().isoformat(),
        }

    def generate(
        self,
        prompt: str,
        compressed_memories: Optional[List[str]] = None,
        goal: Optional[str] = None,
    ) -> str:
        if not self.examples:
            return "No trained local model is available yet. Train a checkpoint before generating."

        context_tokens = _tokenize(prompt)
        for memory in compressed_memories or []:
            context_tokens.extend(_tokenize(memory))
        if goal:
            context_tokens.extend(_tokenize(goal))
        query_vector = _vectorize(context_tokens)

        best_match = max(
            self.examples,
            key=lambda item: _cosine_similarity(query_vector, item["prompt_vector"]),
        )
        memory_count = len(compressed_memories or [])
        return (
            f"{best_match['response']} "
            f"(goal={goal or 'none'}, memories={memory_count}, model={self.model_name})"
        )

    def save_checkpoint(self, checkpoint_path: Optional[str] = None) -> str:
        target = checkpoint_path or self.checkpoint_path
        if not target:
            raise ValueError("checkpoint_path is required to save the local model")
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "saved_at": datetime.now().astimezone().isoformat(),
            "examples": [
                {"prompt": item["prompt"], "response": item["response"]}
                for item in self.examples
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.checkpoint_path = str(path)
        return str(path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.exists():
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.model_name = payload.get("model_name", self.model_name)
        self.train(payload.get("examples", []))
        self.checkpoint_path = str(path)


class ExternalLLMAdapter:
    """
    Bridge external chat APIs with the consciousness-backed local session model.

    When no endpoint is configured or the external call fails, a tiny local
    trainable model is used as a fallback so the integration remains operational
    in local development and GitHub Actions.
    """

    def __init__(
        self,
        model_name: str = "tiny-consciousness",
        endpoint: Optional[str] = None,
        provider: str = "auto",
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        session_id: Optional[str] = None,
        memory_base_dir: str = "consciousness_sessions",
        local_model_path: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.endpoint = endpoint or os.environ.get("MECHAMINER_LLM_ENDPOINT")
        self.provider = provider
        self.api_key = api_key or os.environ.get("MECHAMINER_LLM_API_KEY")
        self.timeout = timeout
        self.session_id = session_id or uuid.uuid4().hex
        self.consciousness_llm = ConsciousnessBackedLLM(
            base_dir=memory_base_dir,
            session_id=self.session_id,
        )
        self.local_model = TinyLocalLanguageModel(
            model_name=model_name,
            checkpoint_path=local_model_path,
        )
        if not self.local_model.examples:
            self.local_model.train(list(DEFAULT_TRAINING_SAMPLES))
            if local_model_path:
                self.local_model.save_checkpoint(local_model_path)

    def train_local(
        self,
        dataset_path: Optional[str] = None,
        samples: Optional[List[Dict[str, str]]] = None,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        training_samples = list(samples or load_training_samples(dataset_path))
        stats = self.local_model.train(training_samples)
        if output_path:
            checkpoint = self.local_model.save_checkpoint(output_path)
            stats["checkpoint_path"] = checkpoint
        return stats

    def _system_prompt(
        self,
        prompt: str,
        goal: Optional[str],
        seed: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        report = seed.get("consciousness_report", {})
        self_model = report.get("self_model", {})
        memories = seed.get("compressed_memories", {}).get("compressed_memories", [])
        payload = {
            "goal": goal,
            "prompt": prompt,
            "mode": self_model.get("mode"),
            "dominant_region": self_model.get("dominant_region"),
            "memories": memories,
            "context": context or {},
        }
        return json.dumps(payload, sort_keys=True)

    def _build_request_body(
        self,
        prompt: str,
        goal: Optional[str],
        seed: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        system_prompt = self._system_prompt(prompt, goal, seed, context)
        user_prompt = prompt if not goal else f"{prompt}\nGoal: {goal}"
        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }

    @staticmethod
    def _extract_external_response(payload: Dict[str, Any]) -> str:
        if not payload:
            return ""
        if isinstance(payload.get("response"), str):
            return payload["response"]
        if isinstance(payload.get("message"), dict):
            content = payload["message"].get("content")
            if isinstance(content, str):
                return content
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            text = choices[0].get("text")
            if isinstance(text, str):
                return text
        return ""

    def _call_external(self, body: Dict[str, Any]) -> str:
        if not self.endpoint:
            return ""

        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            return ""
        return self._extract_external_response(payload)

    def generate(
        self,
        prompt: str,
        goal: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        prefer_external: bool = True,
    ) -> Dict[str, Any]:
        seed = self.consciousness_llm.respond(prompt, goal=goal)
        memories = seed["compressed_memories"]["compressed_memories"]
        body = self._build_request_body(prompt, goal, seed, context)

        provider_used = "local"
        response = ""
        if prefer_external:
            response = self._call_external(body)
            if response:
                provider_used = "external"

        if not response:
            response = self.local_model.generate(prompt, compressed_memories=memories, goal=goal)

        metadata = {"source": "external_llm", "provider": provider_used, "model_name": self.model_name}
        self.consciousness_llm.memory_store.record_thought(response, metadata)
        if self.consciousness_llm._contains_any(response, self.consciousness_llm.MEMORY_PATTERNS):
            self.consciousness_llm.memory_store.record_memory(response, metadata)
        if self.consciousness_llm._contains_any(response, self.consciousness_llm.EMOTION_PATTERNS):
            self.consciousness_llm.memory_store.record_emotion(response, metadata)

        return {
            "provider_used": provider_used,
            "model_name": self.model_name,
            "session_id": self.session_id,
            "response": response,
            "request_body": body,
            "seed_response": seed["response"],
            "compressed_memories": seed["compressed_memories"],
            "storage_paths": seed["storage_paths"],
        }
