"""Small, provider-independent, in-process metric and event registry."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


class MetricsRegistry:
    """Thread-safe aggregates plus a bounded metadata-only event buffer."""

    def __init__(self, max_events: int = 1000) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._observations: dict[str, list[float]] = defaultdict(list)
        self._events: list[dict[str, Any]] = []
        self._max_events = max_events

    def increment(self, name: str, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += 1

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            self._observations[name].append(value)

    def event(self, name: str, fields: Mapping[str, Any]) -> None:
        with self._lock:
            self._events.append({"event": name, **dict(fields)})
            del self._events[: -self._max_events]

    def counter(self, name: str, **labels: str) -> float:
        with self._lock:
            return self._counters[(name, tuple(sorted(labels.items())))]

    def observations(self, name: str) -> list[float]:
        with self._lock:
            return list(self._observations[name])

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)
