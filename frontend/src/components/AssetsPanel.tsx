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
                  {a.kind === "multiview" && <span className="tile-badge">MV</span>}
                  {a.kind === "model3d" && <span className="tile-badge">3D</span>}
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
  if (asset.kind === "model3d") return <Model3DDetails asset={asset} />;
  const mv = asset.kind === "multiview";
  const rows: [string, string | number][] = [
    ...(mv ? ([["Type", pr.mode === "i2mv" ? "Image → multiview" : "Text → multiview"]] as [string, string][]) : []),
    ["Model", pr.model_name ?? pr.model],
    ["Architecture", pr.arch === "sd15" ? "SD 1.5" : "SDXL"],
    [mv ? "View size" : "Size", `${asset.width} × ${asset.height}`],
    ["Seed", asset.seed],
    ["Steps", pr.steps],
    ["CFG", pr.cfg],
    ["Sampler", pr.sampler_label ?? pr.sampler],
    ...(mv
      ? pr.mode === "i2mv"
        ? ([
            ["Background", pr.remove_bg ? "Removed" : "Kept"],
            ["Reference strength", pr.reference_scale ?? 1],
          ] as [string, string | number][])
        : []
      : ([["CLIP skip", pr.clip_skip]] as [string, number][])),
    ["Created", new Date(asset.created_at * 1000).toLocaleString()],
  ];
  return (
    <div className="details">
      <div className="detail-block">
        <div className="section-label">Prompt</div>
        <p className="detail-text">{asset.prompt || "—"}</p>
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

function Model3DDetails({ asset }: { asset: Asset }) {
  const pr = asset.params;
  const rows: [string, string | number][] = [
    ["Engine", pr.textured ? "TripoSR + MV-Adapter texture" : "TripoSR"],
    ...(pr.textured
      ? ([
          ["Texture", `${pr.uv_size} × ${pr.uv_size}`],
          // Textured models from before the baker setting existed were all made with the lean baker.
          ["Baker", pr.baker === "upstream" ? "MV-Adapter (upstream)" : "Lean (Trivexa)"],
          ["Base model", pr.model_name ?? pr.model ?? "—"],
          ["Seed", pr.seed ?? "—"],
        ] as [string, string | number][])
      : []),
    ["Faces", pr.faces?.toLocaleString() ?? "—"],
    ["Vertices", pr.vertices?.toLocaleString() ?? "—"],
    ...(pr.textured
      ? []
      : ([
          ["Mesh resolution", pr.mc_resolution ?? "—"],
          ["Subject fill", pr.foreground_ratio ?? "—"],
        ] as [string, string | number][])),
    ["Background", pr.remove_bg ? "Removed" : "Kept"],
    ["Created", new Date(asset.created_at * 1000).toLocaleString()],
  ];
  return (
    <div className="details">
      {asset.input_url && (
        <div className="detail-block">
          <div className="section-label">{pr.textured ? "Reference" : "Input"}</div>
          <img className="detail-input" src={asset.input_url} alt="Input" />
        </div>
      )}
      {asset.image_urls?.views && (
        <div className="detail-block">
          <div className="section-label">Generated views</div>
          <a href={asset.image_urls.views} target="_blank" rel="noreferrer">
            <img className="detail-input" src={asset.image_urls.views} alt="Views" />
          </a>
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
        <div className="section-label">Files</div>
        <p className="detail-text mono">{asset.model_urls?.glb.replace("/files/", "")}</p>
        <p className="detail-text mono">{asset.model_urls?.obj.replace("/files/", "")}</p>
      </div>
    </div>
  );
}
