import importlib


def _reload_config(monkeypatch, fallback=None, set_fallback=False):
    monkeypatch.setenv("COHERE_PRIMARY_MODEL", "command-a-plus-05-2026")
    if set_fallback:
        monkeypatch.setenv("COHERE_FALLBACK_MODEL", fallback or "")
    else:
        monkeypatch.delenv("COHERE_FALLBACK_MODEL", raising=False)

    import config

    return importlib.reload(config)


def test_duplicate_fallback_is_replaced_with_reasoning(monkeypatch):
    config = _reload_config(
        monkeypatch,
        fallback="command-a-plus-05-2026",
        set_fallback=True,
    )
    assert config.COHERE_PRIMARY_MODEL == "command-a-plus-05-2026"
    assert config.COHERE_FALLBACK_MODEL == "command-a-reasoning-08-2025"


def test_blank_fallback_selects_reasoning_model(monkeypatch):
    config = _reload_config(monkeypatch)
    assert config.COHERE_FALLBACK_MODEL == "command-a-reasoning-08-2025"


def test_distinct_explicit_fallback_is_preserved(monkeypatch):
    config = _reload_config(
        monkeypatch,
        fallback="command-r-plus-08-2024",
        set_fallback=True,
    )
    assert config.COHERE_FALLBACK_MODEL == "command-r-plus-08-2024"


def test_case_insensitive_duplicate_is_replaced(monkeypatch):
    config = _reload_config(
        monkeypatch,
        fallback="COMMAND-A-PLUS-05-2026",
        set_fallback=True,
    )
    assert config.COHERE_FALLBACK_MODEL == "command-a-reasoning-08-2025"
