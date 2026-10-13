import { Images, LayoutGrid, Loader2, SlidersHorizontal, Star } from "lucide-react";
import type { Asset, Job } from "../api";

export type Selection = { kind: "asset" | "job"; id: string } | null;

interface Props {
  tab: "assets" | "details";
  setTab: (t: "assets" | "details") => void;
  assets: Asset[];
  activeJobs: Job[];
  selection: Selection;
  onSelect: (s: Selection) => void;
  favoritesOnly: boolean;
  setFavoritesOnly: (v: boolean) => void;
  onToggleFavorite: (a: Asset) => void;
  hasMore: boolean;
  onLoadMore: () => void;
  detailAsset: Asset | null;
}

export function AssetsPanel(p: Props) {
  return (
    <aside className="panel right-panel">
      <div className="tabs">
        <button className={p.tab === "assets" ? "active" : ""} onClick={() => p.setTab("assets")}>
          <Images size={14} /> Assets
        </button>
        <button className={p.tab === "details" ? "active" : ""} onClick={() => p.setTab("details")}>
          <SlidersHorizontal size={14} /> Details
        </button>
      </div>

      {p.tab === "assets" ? (
        <>
          <div className="filter-row">
            <button className={`chip ${!p.favoritesOnly ? "active" : ""}`} onClick={() => p.setFavoritesOnly(false)} title="All">
              <LayoutGrid size={14} />
            </button>
            <button className={`chip ${p.favoritesOnly ? "active" : ""}`} onClick={() => p.setFavoritesOnly(true)} title="Favorites">
              <Star size={14} />
            </button>
          </div>
          <div className="panel-scroll">
            <div className="asset-grid">
              {p.activeJobs.map((j) => (
                <button
                  key={j.id}
                  className={`tile job-tile ${p.selection?.kind === "job" && p.selection.id === j.id ? "selected" : ""}`}
                  onClick={() => p.onSelect({ kind: "job", id: j.id })}
                >
                  {j.preview ? <img src={j.preview} alt="" /> : <Loader2 className="spin" size={20} />}
                  <span className="tile-badge">
                    {j.status === "running" ? `${Math.round((j.step / Math.max(1, j.total)) * 100)}%` : j.status === "loading" ? "Loading" : "Queued"}
                  </span>
                </button>
              ))}
              {p.assets.map((a) => (
                <div
                  key={a.id}
                  className={`tile ${p.selection?.kind === "asset" && p.selection.id === a.id ? "selected" : ""}`}
                  onClick={() => p.onSelect({ kind: "asset", id: a.id })}
                >
                  <img src={a.thumb_url} alt={a.prompt} loading="lazy" />
                  <button
                    className={`tile-star ${a.favorite ? "on" : ""}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      p.onToggleFavorite(a);
                    }}
                    title="Favorite"
                  >
                    <Star size={13} fill={a.favorite ? "currentColor" : "none"} />
                  </button>
                </div>
              ))}
            </div>
            {p.assets.length === 0 && p.activeJobs.length === 0 && (
              <p className="hint center">{p.favoritesOnly ? "No favorites yet." : "Your generations will appear here."}</p>
            )}
            {p.hasMore && (
              <button className="load-more" onClick={p.onLoadMore}>
                Load more
              </button>
            )}
          </div>
        </>
      ) : (
        <div className="panel-scroll">
          <Details asset={p.detailAsset} />
        </div>
      )}
    </aside>
  );
}

function Details({ asset }: { asset: Asset | null }) {
  if (!asset) return <p className="hint center">Select an asset to see its settings.</p>;
  const pr = asset.params;
  const rows: [string, string | number][] = [
    ["Model", pr.model_name ?? pr.model],
    ["Size", `${asset.width} × ${asset.height}`],
    ["Seed", asset.seed],
    ["Steps", pr.steps],
    ["CFG", pr.cfg],
    ["Sampler", pr.sampler_label ?? pr.sampler],
    ["CLIP skip", pr.clip_skip],
    ["Created", new Date(asset.created_at * 1000).toLocaleString()],
  ];
  return (
    <div className="details">
      <div className="detail-block">
        <div className="section-label">Prompt</div>
        <p className="detail-text">{asset.prompt}</p>
      </div>
      {asset.negative && (
        <div className="detail-block">
          <div className="section-label">Negative prompt</div>
          <p className="detail-text muted">{asset.negative}</p>
        </div>
      )}
      <dl className="detail-table">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
      <div className="detail-block">
        <div className="section-label">File</div>
        <p className="detail-text mono">{asset.file}</p>
      </div>
    </div>
  );
}
