import { Copy, Download, ExternalLink, RotateCcw, Star, Trash2, X } from "lucide-react";
import type { Asset, Job } from "../api";
import { Logo } from "./Logo";

interface Props {
  asset: Asset | null;
  job: Job | null;
  queuePosition: number;
  onCancel: (id: string) => void;
  onToggleFavorite: (a: Asset) => void;
  onDelete: (a: Asset) => void;
  onReuse: (a: Asset) => void;
  notify: (msg: string) => void;
}

export function Viewer({ asset, job, queuePosition, onCancel, onToggleFavorite, onDelete, onReuse, notify }: Props) {
  if (job && job.status !== "done") {
    const { width, height } = job.params;
    const pct = job.total ? Math.round((job.step / job.total) * 100) : 0;
    return (
      <main className="viewer">
        <div className="stage">
          <div className="frame pending" style={{ aspectRatio: `${width} / ${height}` }}>
            {job.preview && <img src={job.preview} alt="" className="preview-img" />}
            <div className="pending-overlay">
              {job.status === "queued" && <span>Queued{queuePosition > 0 ? ` · #${queuePosition}` : ""}</span>}
              {job.status === "loading" && <span>Loading model…</span>}
              {job.status === "running" && (
                <span>
                  Step {job.step} / {job.total}
                </span>
              )}
              {job.status === "error" && <span className="error-text">Failed: {job.error}</span>}
              {job.status === "cancelled" && <span>Cancelled</span>}
            </div>
            {job.status === "running" && (
              <div className="progress">
                <div className="progress-bar" style={{ width: `${pct}%` }} />
              </div>
            )}
          </div>
        </div>
        <div className="toolbar">
          <p className="toolbar-prompt" title={job.params.prompt}>
            {job.params.prompt}
          </p>
          {(job.status === "queued" || job.status === "loading" || job.status === "running") && (
            <button className="tool-btn" onClick={() => onCancel(job.id)}>
              <X size={15} /> Cancel
            </button>
          )}
        </div>
      </main>
    );
  }

  if (!asset) {
    return (
      <main className="viewer">
        <div className="empty-state">
          <Logo size={44} />
          <h1>Ready for a new image?</h1>
          <p>Describe it on the left and press Generate. Everything is saved locally.</p>
        </div>
      </main>
    );
  }

  const copy = async (text: string, what: string) => {
    try {
      await navigator.clipboard.writeText(text);
      notify(`${what} copied`);
    } catch {
      notify("Clipboard not available");
    }
  };

  return (
    <main className="viewer">
      <div className="stage">
        <a href={asset.url} target="_blank" rel="noreferrer" className="frame" style={{ aspectRatio: `${asset.width} / ${asset.height}` }}>
          <img src={asset.url} alt={asset.prompt} />
        </a>
      </div>
      <div className="toolbar">
        <p className="toolbar-prompt" title={asset.prompt}>
          {asset.prompt}
        </p>
        <div className="toolbar-actions">
          <button className={`tool-btn ${asset.favorite ? "fav" : ""}`} onClick={() => onToggleFavorite(asset)} title="Favorite">
            <Star size={15} fill={asset.favorite ? "currentColor" : "none"} />
          </button>
          <button className="tool-btn" onClick={() => copy(asset.prompt, "Prompt")} title="Copy prompt">
            <Copy size={15} />
          </button>
          <button className="tool-btn" onClick={() => onReuse(asset)} title="Load these settings into the panel">
            <RotateCcw size={15} /> Reuse
          </button>
          <a className="tool-btn" href={asset.url} download title="Download PNG">
            <Download size={15} />
          </a>
          <a className="tool-btn" href={asset.url} target="_blank" rel="noreferrer" title="Open full size">
            <ExternalLink size={15} />
          </a>
          <button className="tool-btn danger" onClick={() => onDelete(asset)} title="Delete">
            <Trash2 size={15} />
          </button>
        </div>
      </div>
    </main>
  );
}
