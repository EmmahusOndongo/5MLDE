from anime_mlops.api.cache import PredictionCache


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def build_cache(clock, *, enabled=True, max_entries=2, ttl_seconds=10):
    return PredictionCache(
        enabled=enabled,
        max_entries=max_entries,
        ttl_seconds=ttl_seconds,
        clock=clock,
    )


def test_cache_expires_entries_and_isolates_model_versions():
    clock = Clock()
    cache = build_cache(clock)
    key = cache.key([{"num_episodes": 12, "genres": ["Action"]}])

    cache.put(key, [7.1], model_version="1")

    assert cache.get(key, model_version="1") == (7.1,)
    assert cache.get(key, model_version="2") is None

    clock.now = 10.0
    assert cache.get(key, model_version="1") is None


def test_cache_is_lru_bounded():
    clock = Clock()
    cache = build_cache(clock, max_entries=2)

    cache.put("a", [1.0], model_version="1")
    cache.put("b", [2.0], model_version="1")
    assert cache.get("a", model_version="1") == (1.0,)
    cache.put("c", [3.0], model_version="1")

    assert cache.get("a", model_version="1") == (1.0,)
    assert cache.get("b", model_version="1") is None
    assert cache.get("c", model_version="1") == (3.0,)


def test_disabled_cache_never_stores_predictions():
    cache = build_cache(Clock(), enabled=False)
    cache.put("key", [7.0], model_version="1")
    assert cache.get("key", model_version="1") is None
