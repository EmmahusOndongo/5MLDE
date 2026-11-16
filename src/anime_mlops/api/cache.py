"""Cache mémoire LRU/TTL des prédictions, isolé par version du modèle."""

import json
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from time import monotonic


@dataclass(frozen=True)
class CacheEntry:
    predictions: tuple[float, ...]
    expires_at: float


class PredictionCache:
    """Cache local borné ; une instance est utilisée par worker API."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_entries: int,
        ttl_seconds: float,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.enabled = enabled
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._lock = RLock()

    @staticmethod
    def key(records: list[dict[str, object]]) -> str:
        serialized = json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        return sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, key: str, *, model_version: str) -> tuple[float, ...] | None:
        if not self.enabled:
            return None
        composite_key = (model_version, key)
        now = self._clock()
        with self._lock:
            entry = self._entries.get(composite_key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                del self._entries[composite_key]
                return None
            self._entries.move_to_end(composite_key)
            return entry.predictions

    def put(self, key: str, predictions: list[float], *, model_version: str) -> None:
        if not self.enabled:
            return
        now = self._clock()
        composite_key = (model_version, key)
        with self._lock:
            self._remove_expired(now)
            self._entries[composite_key] = CacheEntry(
                predictions=tuple(predictions),
                expires_at=now + self.ttl_seconds,
            )
            self._entries.move_to_end(composite_key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def _remove_expired(self, now: float) -> None:
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            del self._entries[key]
