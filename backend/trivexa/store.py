"""Local persistence: generated files under data/outputs and their records in SQLite."""

import json
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from .config import DB_PATH, OUTPUTS_DIR

THUMB_SIZE = 384

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    file        TEXT NOT NULL,
    thumb       TEXT,
    width       INTEGER,
    height      INTEGER,
    prompt      TEXT,
    negative    TEXT,
    model       TEXT,
    seed        INTEGER,
    params      TEXT NOT NULL DEFAULT '{}',
    favorite    INTEGER NOT NULL DEFAULT 0,
    parent_id   TEXT
);
CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at DESC);
"""


class Store:
    def __init__(self, db_path: Path = DB_PATH):
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    @staticmethod
    def _row_to_asset(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        d["favorite"] = bool(d["favorite"])
        d["url"] = f"/files/{d['file']}"
        d["thumb_url"] = f"/files/{d['thumb']}" if d["thumb"] else d["url"]
        return d

    def save_image(self, image: Image.Image, params: dict) -> dict:
        asset_id = uuid.uuid4().hex
        now = time.time()
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        stem = f"{datetime.fromtimestamp(now).strftime('%H%M%S')}_{params.get('seed', 0)}_{asset_id[:8]}"

        rel_file = Path("images") / day / f"{stem}.png"
        rel_thumb = Path("thumbs") / day / f"{stem}.webp"
        (OUTPUTS_DIR / rel_file).parent.mkdir(parents=True, exist_ok=True)
        (OUTPUTS_DIR / rel_thumb).parent.mkdir(parents=True, exist_ok=True)

        info = PngInfo()
        info.add_text("parameters", _a1111_parameters(params))
        info.add_text("trivexa", json.dumps(params))
        image.save(OUTPUTS_DIR / rel_file, pnginfo=info)

        thumb = image.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(OUTPUTS_DIR / rel_thumb, quality=85)

        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO assets (id, kind, created_at, file, thumb, width, height, prompt, negative, model, seed, params)"
                " VALUES (?, 'image', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset_id, now, rel_file.as_posix(), rel_thumb.as_posix(), image.width, image.height,
                    params.get("prompt"), params.get("negative"), params.get("model"), params.get("seed"),
                    json.dumps(params),
                ),
            )
        return self.get(asset_id)

    def get(self, asset_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()
        return self._row_to_asset(row) if row else None

    def list(self, kind: str | None = None, favorite: bool | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        where, args = [], []
        if kind:
            where.append("kind = ?")
            args.append(kind)
        if favorite is not None:
            where.append("favorite = ?")
            args.append(int(favorite))
        sql = "SELECT * FROM assets"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._db.execute(sql, (*args, limit, offset)).fetchall()
        return [self._row_to_asset(r) for r in rows]

    def set_favorite(self, asset_id: str, favorite: bool) -> dict | None:
        with self._lock, self._db:
            self._db.execute("UPDATE assets SET favorite = ? WHERE id = ?", (int(favorite), asset_id))
        return self.get(asset_id)

    def delete(self, asset_id: str) -> bool:
        asset = self.get(asset_id)
        if not asset:
            return False
        for rel in (asset["file"], asset["thumb"]):
            if rel:
                (OUTPUTS_DIR / rel).unlink(missing_ok=True)
        with self._lock, self._db:
            self._db.execute("DELETE FROM assets WHERE id = ?", (asset_id,))
        return True


def _a1111_parameters(p: dict) -> str:
    """The de-facto 'parameters' PNG text chunk, so other tools can read the settings back."""
    lines = [p.get("prompt", "")]
    if p.get("negative"):
        lines.append(f"Negative prompt: {p['negative']}")
    lines.append(
        f"Steps: {p.get('steps')}, Sampler: {p.get('sampler_label', p.get('sampler'))}, CFG scale: {p.get('cfg')}, "
        f"Seed: {p.get('seed')}, Size: {p.get('width')}x{p.get('height')}, Model: {p.get('model_name', p.get('model'))}"
    )
    return "\n".join(lines)
