"""Keep the databases on hosts whose disk is wiped on every restart (e.g. Render's free plan).

With OUTHIT_BACKUP_REPO (a private Hugging Face dataset, e.g. "you/outhit-data") and HF_TOKEN
set, combined.py restores every database from the latest backup before the services start,
and uploads a fresh one every OUTHIT_BACKUP_MINUTES and on shutdown. Without them, nothing
here runs and the databases simply live on the local disk.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import tarfile
import tempfile
from pathlib import Path

log = logging.getLogger("outhit.backup")

REPO = os.environ.get("OUTHIT_BACKUP_REPO", "")
INTERVAL_MINUTES = int(os.environ.get("OUTHIT_BACKUP_MINUTES", "5"))
ARCHIVE = "outhit-data.tar.gz"


def enabled() -> bool:
    return bool(REPO and os.environ.get("HF_TOKEN"))


def restore(data_dir: Path) -> None:
    """Unpack the last backup into an empty data directory (a fresh container)."""
    if not enabled() or any(data_dir.glob("*.db")):
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(REPO, ARCHIVE, repo_type="dataset")
    except (EntryNotFoundError, RepositoryNotFoundError):
        log.info("no backup in %s yet; starting with empty databases", REPO)
        return
    with tarfile.open(path) as tar:
        tar.extractall(data_dir, filter="data")
    log.info("restored %s from %s", ", ".join(sorted(p.name for p in data_dir.glob("*.db"))), REPO)


def backup(data_dir: Path) -> None:
    """Upload a consistent copy of every database, keeping only the latest backup."""
    from huggingface_hub import HfApi

    databases = sorted(data_dir.glob("*.db"))
    if not databases:
        return
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp, ARCHIVE)
        with tarfile.open(packed, "w:gz") as tar:
            for db in databases:
                snapshot = Path(tmp, db.name)
                with sqlite3.connect(db) as live, sqlite3.connect(snapshot) as copy:
                    live.backup(copy)   # safe while the services keep writing
                tar.add(snapshot, arcname=db.name)
        api.upload_file(path_or_fileobj=str(packed), path_in_repo=ARCHIVE, repo_id=REPO,
                        repo_type="dataset", commit_message="backup")
    api.super_squash_history(REPO, repo_type="dataset")


async def run_forever(data_dir: Path, stop: asyncio.Event) -> None:
    if not enabled():
        return
    log.info("backing up %s to %s every %s min", data_dir, REPO, INTERVAL_MINUTES)
    while True:
        try:
            await asyncio.wait_for(stop.wait(), timeout=INTERVAL_MINUTES * 60)
        except asyncio.TimeoutError:
            pass
        try:
            await asyncio.to_thread(backup, data_dir)
        except Exception:
            log.exception("backup failed")
        if stop.is_set():
            return
