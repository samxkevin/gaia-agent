import importlib


def test_fallback_model_is_distinct_from_primary(monkeypatch):
    monkeypatch.setenv("COHERE_PRIMARY_MODEL", "command-a-plus-05-2026")
    monkeypatch.setenv("COHERE_FALLBACK_MODEL", "command-a-plus-05-2026")

    import config

    config = importlib.reload(config)
    assert config.COHERE_PRIMARY_MODEL != config.COHERE_FALLBACK_MODEL
    assert config.COHERE_FALLBACK_MODEL == "command-a-reasoning-08-2025"


def test_blank_fallback_selects_reasoning_model(monkeypatch):
    monkeypatch.setenv("COHERE_PRIMARY_MODEL", "command-a-plus-05-2026")
    monkeypatch.delenv("COHERE_FALLBACK_MODEL", raising=False)

    import config

    config = importlib.reload(config)
    assert config.COHERE_FALLBACK_MODEL == "command-a-reasoning-08-2025"
