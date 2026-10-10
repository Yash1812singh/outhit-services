import os
import shutil
import tempfile

# Must run before any research module is imported (settings are lru_cached).
_TMP = tempfile.mkdtemp(prefix="outhit-research-test-")
os.environ["RESEARCH_DATABASE_URL"] = f"sqlite:///{_TMP}/research.db"
os.environ["RESEARCH_ENABLE_SCHEDULER"] = "false"

import pytest  # noqa: E402

from research.db import Base, engine, init_db  # noqa: E402


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
