"""One-off: copy data from the old monolith databases into the per-service databases.

    .venv/bin/python scripts/migrate_legacy.py [--backend ../outhit-backend/outhit.db] [--ai ../outhit-ai/outhit_ai.db]

Safe to re-run: rows whose primary key already exists are skipped.
"""

import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (source, table) -> (service dir, target db)
PLAN = [
    ("backend", "users", "gateway", "auth.db"),
    ("backend", "financial_profiles", "advisor", "advisor.db"),
    ("backend", "stock_quotes", "market_data", "market_data.db"),
    ("backend", "market_refreshes", "market_data", "market_data.db"),
    ("backend", "fundamentals_cache", "market_data", "market_data.db"),
    ("ai", "companies", "intelligence", "intelligence.db"),
    ("ai", "news_items", "intelligence", "intelligence.db"),
    ("ai", "predictions", "intelligence", "intelligence.db"),
    ("ai", "recommendation_log", "research", "research.db"),
]
CREATE_TABLES = {
    "gateway": "import gateway.auth as m; m.Base.metadata.create_all(m.engine)",
    "advisor": "import advisor.db as m; m.Base.metadata.create_all(m.engine)",
    "market_data": "import market_data.models; import market_data.db as m; m.Base.metadata.create_all(m.engine)",
    "intelligence": "import intelligence.db as m; m.init_db()",
    "research": "import research.db as m; m.init_db()",
}


def columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=str(ROOT.parent / "outhit-backend" / "outhit.db"))
    ap.add_argument("--ai", default=str(ROOT.parent / "outhit-ai" / "outhit_ai.db"))
    args = ap.parse_args()
    sources = {"backend": Path(args.backend), "ai": Path(args.ai)}

    for service, code in CREATE_TABLES.items():
        (ROOT / service / "data").mkdir(exist_ok=True)
        subprocess.run([sys.executable, "-c", code], cwd=ROOT / service, env={"PYTHONPATH": str(ROOT)}, check=True)

    for src_name, table, service, db_name in PLAN:
        src_path = sources[src_name]
        if not src_path.exists():
            print(f"skip {table}: {src_path} not found")
            continue
        src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
        dst = sqlite3.connect(ROOT / service / "data" / db_name)
        try:
            src_cols = columns(src, table)
            if not src_cols:
                print(f"skip {table}: not in {src_path.name}")
                continue
            shared = [c for c in src_cols if c in set(columns(dst, table))]
            rows = src.execute(f"SELECT {', '.join(shared)} FROM {table}").fetchall()
            before = dst.total_changes
            dst.executemany(f"INSERT OR IGNORE INTO {table} ({', '.join(shared)}) VALUES ({', '.join('?' * len(shared))})", rows)
            dst.commit()
            print(f"{table:20s} -> {service}/{db_name}: {dst.total_changes - before} of {len(rows)} rows copied")
        finally:
            src.close()
            dst.close()


if __name__ == "__main__":
    main()
