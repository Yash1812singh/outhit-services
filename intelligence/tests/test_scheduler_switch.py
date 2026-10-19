import threading

from intelligence import scheduler


def test_disabled_loops_never_start(monkeypatch):
    monkeypatch.setattr(scheduler, "DISABLED", {"news"})
    before = threading.active_count()
    scheduler._every(60, lambda: None, "news")
    assert threading.active_count() == before
