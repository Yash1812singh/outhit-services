import os
import shutil
import tempfile

# Must run before any intelligence module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-intelligence-test-")
os.environ["INTELLIGENCE_DATABASE_URL"] = f"sqlite:///{_TMP}/intelligence.db"
os.environ["INTELLIGENCE_ENABLE_SCHEDULER"] = "false"
os.environ["INTELLIGENCE_GROQ_API_KEY"] = ""

import pytest  # noqa: E402

from intelligence.db import Base, engine, init_db  # noqa: E402

INTERNAL = {"X-Internal-Token": "test-internal-token"}


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(bind=engine)
    init_db()
    yield


@pytest.fixture(scope="session", autouse=True)
def _cleanup():
    yield
    engine.dispose()
    shutil.rmtree(_TMP, ignore_errors=True)
