from research import jobs


def test_disabled_jobs_answer_without_running(monkeypatch):
    ran = []
    monkeypatch.setattr(jobs, "DISABLED", ("tournament", "trade:bt"))
    out = jobs.run("tournament:in", lambda progress: ran.append(1), ttl=60)
    assert out["status"] == "error" and "memory" in out["message"]
    assert ran == [] and "tournament:in" not in jobs._jobs
