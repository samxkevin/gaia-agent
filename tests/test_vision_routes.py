import models
from models import (
    CohereFailoverClient,
    CohereTextFailoverClient,
    CohereVerificationClient,
    Route,
)


def test_multimodal_routes_use_only_dedicated_vision_model(monkeypatch):
    monkeypatch.setattr(models, "COHERE_PRIMARY_API_KEY", "key-1")
    monkeypatch.setattr(models, "COHERE_FALLBACK_API_KEY", "key-2")
    monkeypatch.setattr(models, "COHERE_VISION_MODEL", "command-a-vision-07-2025")
    expected = [
        Route("primary_key", "command-a-vision-07-2025", "key-1"),
        Route("fallback_key", "command-a-vision-07-2025", "key-2"),
    ]
    assert models.get_vision_routes() == expected
    assert CohereFailoverClient().routes == expected
    assert all("reasoning" not in route.model_id for route in expected)


def test_text_routes_remain_unchanged(monkeypatch):
    routes = [
        Route("primary_key", "command-a-plus-05-2026", "key-1"),
        Route("fallback_key", "command-a-plus-05-2026", "key-2"),
        Route("primary_key", "command-a-reasoning-08-2025", "key-1"),
        Route("fallback_key", "command-a-reasoning-08-2025", "key-2"),
    ]
    monkeypatch.setattr(models, "get_chat_routes", lambda: routes)
    assert models.get_chat_routes() == routes
    assert CohereTextFailoverClient().routes == routes
    verification = CohereVerificationClient().routes
    assert verification == routes[:2]
    assert all("reasoning" not in route.model_id for route in verification)
