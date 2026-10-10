from types import SimpleNamespace as NS

from intelligence import llm_news


# ---------------- LLM news ----------------

def test_llm_batch_parsing(monkeypatch):
    content = '{"scores":[{"i":1,"score":0.8},{"i":2,"score":-2}]}'
    fake = NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[NS(message=NS(content=content))]))))
    monkeypatch.setattr(llm_news, "_client", lambda: fake)
    out = llm_news.score_batch([("A Ltd", "A beats estimates"), ("B Ltd", "B fraud probe"), ("C Ltd", "C results date")])
    assert out == [0.8, -1.0, None]  # clipped to [-1, 1]; missing -> None
