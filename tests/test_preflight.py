from evaluation import preflight


def test_javascript_runtime_preflight_reports_available_runtime(monkeypatch):
    monkeypatch.setattr(preflight, "find_javascript_runtime", lambda: ("node", "/usr/bin/node"))
    status, error = preflight.javascript_runtime_status()
    assert status == {"name": "node", "path": "/usr/bin/node"}
    assert error is None


def test_javascript_runtime_preflight_reports_clear_failure(monkeypatch):
    monkeypatch.setattr(preflight, "find_javascript_runtime", lambda: None)
    status, error = preflight.javascript_runtime_status()
    assert status is None
    assert "deno, node, quickjs, or bun" in error
