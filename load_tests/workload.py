"""Jeu de requêtes déterministe utilisé par Locust."""

from copy import deepcopy

Payload = dict[str, object]

BASE_PAYLOAD: Payload = {
    "num_episodes": 12,
    "average_episode_duration": 1_440.0,
    "start_year": 2027,
    "prequel_score": 7.2,
    "media_type": "tv",
    "rating": "pg_13",
    "source": "MANGA",
    "start_season": "spring",
    "start_weekday": "monday",
    "is_sequel": False,
    "genres": ["Action", "Drama"],
    "studios": ["Bones"],
    "tags": ["Shounen"],
}


def _build_hot_payloads() -> tuple[Payload, ...]:
    payloads = []
    seasons = ("winter", "spring", "summer", "fall")
    weekdays = ("monday", "tuesday", "wednesday", "thursday")
    for index in range(8):
        payload = deepcopy(BASE_PAYLOAD)
        payload.update(
            {
                "num_episodes": 12 + index,
                "average_episode_duration": 1_380.0 + index * 15,
                "start_year": 2025 + index % 4,
                "prequel_score": 6.5 + index * 0.25,
                "start_season": seasons[index % len(seasons)],
                "start_weekday": weekdays[index % len(weekdays)],
                "is_sequel": index % 2 == 0,
            }
        )
        payloads.append(payload)
    return tuple(payloads)


HOT_PAYLOADS = _build_hot_payloads()


def hot_payload(index: int) -> Payload:
    """Retourne une copie d'une entrée du petit ensemble récurrent."""
    return deepcopy(HOT_PAYLOADS[index % len(HOT_PAYLOADS)])


def cold_payload(index: int) -> Payload:
    """Retourne une entrée unique sans introduire de catégorie inconnue au modèle."""
    payload = deepcopy(BASE_PAYLOAD)
    # Un score numérique à six décimales produit neuf millions de clés valides et distinctes.
    payload["prequel_score"] = round(0.5 + (index % 9_000_000) / 1_000_000, 6)
    return payload
