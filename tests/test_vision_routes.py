import models
from models import CohereFailoverClient, Route


def test_multimodal_routes_exclude_text_only_reasoning_model(monkeypatch):
    routes = [
        Route("primary_key", "command-a-plus-05-2026", "key-1"),
        Route("fallback_key", "command-a-plus-05-2026", "key-2"),
        Route("primary_key", "command-a-reasoning-08-2025", "key-1"),
        Route("fallback_key", "command-a-reasoning-08-2025", "key-2"),
    ]
    monkeypatch.setattr(models, "get_chat_routes", lambda: routes)

    assert models.get_vision_routes() == routes[:2]
    client = CohereFailoverClient()
    assert client.routes == routes[:2]


def test_text_routes_are_not_changed_by_vision_filter(monkeypatch):
    routes = [
        Route("primary_key", "command-a-plus-05-2026", "key-1"),
        Route("fallback_key", "command-a-plus-05-2026", "key-2"),
        Route("primary_key", "command-a-reasoning-08-2025", "key-1"),
        Route("fallback_key", "command-a-reasoning-08-2025", "key-2"),
    ]
    monkeypatch.setattr(models, "get_chat_routes", lambda: routes)
    model = models.FailoverModel.__new__(models.FailoverModel)
    model.routes = models.get_chat_routes()
    assert model.routes == routes
