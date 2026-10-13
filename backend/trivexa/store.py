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
    parent_id   TEXT,
    extra_files TEXT NOT NULL DEFAULT '{}'
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
            columns = {r["name"] for r in self._db.execute("PRAGMA table_info(assets)")}
            if "extra_files" not in columns:  # added with multi-view assets
                self._db.execute("ALTER TABLE assets ADD COLUMN extra_files TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _row_to_asset(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d["params"] or "{}")
        d["favorite"] = bool(d["favorite"])
        d["url"] = f"/files/{d['file']}"
        d["thumb_url"] = f"/files/{d['thumb']}" if d["thumb"] else d["url"]
        extra = json.loads(d.pop("extra_files") or "{}")
        if "views" in extra:
            d["views"] = [{"azimuth": v["azimuth"], "url": f"/files/{v['file']}"} for v in extra["views"]]
        if "reference" in extra:
            d["reference_url"] = f"/files/{extra['reference']}"
        if "model" in extra:
            d["model_urls"] = {fmt: f"/files/{rel}" for fmt, rel in extra["model"].items()}
            d["input_url"] = f"/files/{extra['input']}"
            d["thumb_rendered"] = extra.get("thumb_rendered", False)
            if d["thumb_rendered"]:
                d["thumb_url"] += "?rendered"  # same file, new content: bust the browser cache
        return d

    @staticmethod
    def _new_stem(params: dict) -> tuple[str, float, str, str]:
        asset_id = uuid.uuid4().hex
        now = time.time()
        day = datetime.fromtimestamp(now).strftime("%Y-%m-%d")
        stem = f"{datetime.fromtimestamp(now).strftime('%H%M%S')}_{params.get('seed', 0)}_{asset_id[:8]}"
        return asset_id, now, day, stem

    @staticmethod
    def _save_thumb(image: Image.Image, rel_thumb: Path) -> None:
        (OUTPUTS_DIR / rel_thumb).parent.mkdir(parents=True, exist_ok=True)
        thumb = image.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
        thumb.save(OUTPUTS_DIR / rel_thumb, quality=85)

    def _insert(self, asset_id, kind, now, rel_file, rel_thumb, width, height, params, parent_id=None, extra=None) -> dict:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO assets (id, kind, created_at, file, thumb, width, height, prompt, negative, model, seed,"
                " params, parent_id, extra_files) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    asset_id, kind, now, rel_file.as_posix(), rel_thumb.as_posix(), width, height,
                    params.get("prompt"), params.get("negative"), params.get("model"), params.get("seed"),
                    json.dumps(params), parent_id, json.dumps(extra or {}),
                ),
            )
        return self.get(asset_id)

    def save_image(self, image: Image.Image, params: dict) -> dict:
        asset_id, now, day, stem = self._new_stem(params)
        rel_file = Path("images") / day / f"{stem}.png"
        rel_thumb = Path("thumbs") / day / f"{stem}.webp"
        (OUTPUTS_DIR / rel_file).parent.mkdir(parents=True, exist_ok=True)

        info = PngInfo()
        info.add_text("parameters", _a1111_parameters(params))
        info.add_text("trivexa", json.dumps(params))
        image.save(OUTPUTS_DIR / rel_file, pnginfo=info)
        self._save_thumb(image, rel_thumb)
        return self._insert(asset_id, "image", now, rel_file, rel_thumb, image.width, image.height, params)

    def save_multiview(
        self,
        views: list[Image.Image],
        azimuths: list[int],
        params: dict,
        reference: Image.Image | None = None,
        parent_id: str | None = None,
        grid_cols: int = 3,
    ) -> dict:
        """Each view as its own PNG plus a grid image (the asset's main file) in one folder per generation."""
        asset_id, now, day, stem = self._new_stem(params)
        rel_dir = Path("multiview") / day / stem
        (OUTPUTS_DIR / rel_dir).mkdir(parents=True, exist_ok=True)

        info = PngInfo()
        info.add_text("trivexa", json.dumps(params))
        extra: dict = {"views": []}
        for az, view in zip(azimuths, views):
            rel = rel_dir / f"view_{az:03d}.png"
            view.save(OUTPUTS_DIR / rel, pnginfo=info)
            extra["views"].append({"azimuth": az, "file": rel.as_posix()})
        if reference is not None:
            rel = rel_dir / "reference.png"
            reference.save(OUTPUTS_DIR / rel)
            extra["reference"] = rel.as_posix()

        w, h = views[0].size
        rows = -(-len(views) // grid_cols)
        grid = Image.new("RGB", (w * grid_cols, h * rows), "white")
        for i, view in enumerate(views):
            grid.paste(view, ((i % grid_cols) * w, (i // grid_cols) * h))
        rel_file = rel_dir / "grid.png"
        grid.save(OUTPUTS_DIR / rel_file, pnginfo=info)
        rel_thumb = Path("thumbs") / day / f"{stem}_mv.webp"
        self._save_thumb(grid, rel_thumb)
        return self._insert(asset_id, "multiview", now, rel_file, rel_thumb, w, h, params, parent_id, extra)

    def save_model3d(self, mesh, input_image: Image.Image, params: dict, parent_id: str | None = None) -> dict:
        """GLB (for the viewer) and OBJ (for DCC tools) plus the processed input, one folder per model.

        The thumbnail starts as the input image; the UI replaces it with a render of the model (set_thumbnail).
        """
        asset_id, now, day, stem = self._new_stem(params)
        rel_dir = Path("models3d") / day / stem
        (OUTPUTS_DIR / rel_dir).mkdir(parents=True, exist_ok=True)
        files = {"glb": rel_dir / "model.glb", "obj": rel_dir / "model.obj"}
        for fmt, rel in files.items():
            mesh.export(OUTPUTS_DIR / rel)
        rel_input = rel_dir / "input.png"
        input_image.save(OUTPUTS_DIR / rel_input)
        rel_thumb = Path("thumbs") / day / f"{stem}_3d.webp"
        self._save_thumb(input_image, rel_thumb)
        extra = {"model": {fmt: rel.as_posix() for fmt, rel in files.items()}, "input": rel_input.as_posix()}
        params = params | {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
        return self._insert(asset_id, "model3d", now, files["glb"], rel_thumb, input_image.width, input_image.height,
                            params, parent_id, extra)

    def set_thumbnail(self, asset_id: str, image: Image.Image) -> dict | None:
        asset = self.get(asset_id)
        if not asset:
            return None
        with self._lock:
            row = self._db.execute("SELECT thumb, extra_files FROM assets WHERE id = ?", (asset_id,)).fetchone()
        rel_thumb = Path(row["thumb"])
        self._save_thumb(image.convert("RGB"), rel_thumb)
        extra = json.loads(row["extra_files"] or "{}") | {"thumb_rendered": True}
        with self._lock, self._db:
            self._db.execute("UPDATE assets SET extra_files = ? WHERE id = ?", (json.dumps(extra), asset_id))
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
        if asset["kind"] in ("multiview", "model3d"):
            folder = (OUTPUTS_DIR / asset["file"]).parent
            for f in folder.iterdir():
                if f.is_file():
                    f.unlink()
            folder.rmdir()
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
