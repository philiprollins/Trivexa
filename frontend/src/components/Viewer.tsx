import { Anchor, Bone, Box, Copy, Download, ExternalLink, Footprints, Layers, PaintBucket, Pause, PersonStanding, Play, Puzzle, Pyramid, RotateCcw, Rotate3d, Star, Trash2, X } from "lucide-react";
import { lazy, Suspense, useEffect, useState } from "react";
import type { Asset, Job } from "../api";
import { Segmented } from "./controls";
import { Logo } from "./Logo";
import type { RenderMode } from "./ModelViewer";

// three.js is large; load it only when a 3D model is first shown.
const ModelViewer = lazy(() => import("./ModelViewer").then((m) => ({ default: m.ModelViewer })));

interface Props {
  asset: Asset | null;
  job: Job | null;
  queuePosition: number;
  onCancel: (id: string) => void;
  onToggleFavorite: (a: Asset) => void;
  onDelete: (a: Asset) => void;
  onReuse: (a: Asset) => void;
  onMakeMultiview: (a: Asset) => void;
  onMake3D: (a: Asset) => void;
  onTexture: (a: Asset) => void;
  onRemesh: (a: Asset) => void;
  onRig: (a: Asset) => void;
  onAnimate: (a: Asset) => void;
  onSegment: (a: Asset) => void;
  onModelSnapshot: (a: Asset, dataUrl: string) => void;
  notify: (msg: string) => void;
}

const VIEW_NAMES: Record<number, string> = { 0: "Front", 90: "Right", 180: "Back", 270: "Left" };

const modelLabel = (p: Asset["params"]) =>
  p.segmented
    ? `Segmented model · ${p.parts} parts`
    : p.animated
    ? `Animated model · ${p.prompt}`
    : p.rigged ? `Rigged model · ${p.bones} bones` : p.textured ? "Textured model" : p.remeshed ? "Remeshed model" : "3D model";

export function Viewer(props: Props) {
  const { asset, job, queuePosition, onCancel, onToggleFavorite, onDelete, onReuse, onMakeMultiview, onMake3D, onTexture, onRemesh, onRig, onAnimate, onSegment, notify } = props;
  const [renderMode, setRenderMode] = useState<RenderMode>("color");
  const [autoRotate, setAutoRotate] = useState(true);
  const [showBones, setShowBones] = useState(true);
  const [posing, setPosing] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [inPlace, setInPlace] = useState(true);
  // The open model's animation clips and the one playing (undefined: the latest, i.e. the last in the GLB).
  const [clipNames, setClipNames] = useState<string[]>([]);
  const [clip, setClip] = useState<number | undefined>(undefined);
  useEffect(() => {
    setClipNames([]);
    setClip(undefined);
  }, [asset?.id]);
  // A segmented model opens on its parts; other models fall back to Color if Parts was left on.
  const segmented = !!asset?.params.segmented;
  useEffect(() => {
    if (segmented) setRenderMode("parts");
  }, [asset?.id, segmented]);

  if (job && job.status !== "done") {
    const [width, height] =
      job.kind === "multiview" || job.kind === "texture" ? [3, 2] : job.kind === "model3d" || job.kind === "remesh" || job.kind === "rig" || job.kind === "animate" || job.kind === "segment" ? [1, 1] : [job.params.width, job.params.height];
    const pct = job.total ? Math.round((job.step / job.total) * 100) : 0;
    return (
      <main className="viewer">
        <div className="stage">
          <div className="frame pending" style={{ aspectRatio: `${width} / ${height}` }}>
            {job.preview && <img src={job.preview} alt="" className="preview-img" />}
            <div className="pending-overlay">
              {job.status === "queued" && <span>Queued{queuePosition > 0 ? ` · #${queuePosition}` : ""}</span>}
              {job.status === "loading" && (
                <span>{{ image: "Loading model…", texture: "Unwrapping UVs & preparing…", remesh: "Loading MeshFlow…", rig: "Loading SkinTokens…", animate: "Loading UniMate…", segment: "Loading PartField…" }[job.kind as string] ?? "Preparing…"}</span>
              )}
              {job.status === "running" && (
                <span>{job.kind === "model3d"
                    ? ["Encoding image…", "Extracting mesh…", "Saving…"][job.step - 1]
                    : job.kind === "rig"
                    ? `${job.step < 40 ? "Predicting skeleton" : "Predicting skin weights"} · ${job.step}%`
                    : job.kind === "segment"
                    ? job.step < 40 ? "Predicting part features…" : "Clustering parts…"
                    : job.kind === "animate"
                    ? `Generating motion · ${job.step}%`
                    : `Step ${job.step} / ${job.total}`}</span>
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
            {job.kind === "texture"
              ? "Texturing 3D model"
              : job.kind === "remesh"
              ? "Remeshing 3D model"
              : job.kind === "rig"
              ? "Rigging 3D model"
              : job.kind === "segment"
              ? "Segmenting 3D model"
              : job.kind === "animate"
              ? `Animating: ${job.params.prompt}`
              : job.params.prompt || (job.kind === "multiview" ? "Multiview from image" : job.kind === "model3d" ? (job.params.source_views ? `3D model from ${job.params.source_views.length} views` : "3D model from image") : "")}
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
        {asset.kind === "model3d" && asset.model_urls ? (
          <div className="frame model-frame">
            <Suspense fallback={null}>
              <ModelViewer
                url={asset.model_urls.glb}
                mode={renderMode === "parts" && !asset.params.segmented ? "color" : renderMode}
                autoRotate={autoRotate}
                skeleton={!!asset.params.rigged && showBones}
                pose={!!asset.params.rigged && !asset.params.animated && posing}
                play={!!asset.params.animated && playing}
                inPlace={inPlace}
                clip={clip}
                onClips={setClipNames}
                onSnapshot={asset.thumb_rendered ? undefined : (dataUrl) => props.onModelSnapshot(asset, dataUrl)}
              />
            </Suspense>
            <div className="model-controls">
              <Segmented
                options={[
                  { value: "color", label: "Color" },
                  ...(asset.params.segmented ? [{ value: "parts" as const, label: "Parts" }] : []),
                  { value: "clay", label: "Clay" },
                  { value: "wireframe", label: "Wire" },
                ]}
                value={renderMode === "parts" && !asset.params.segmented ? "color" : renderMode}
                onChange={setRenderMode}
              />
              <button
                className={`icon-btn ${autoRotate ? "active" : ""}`}
                onClick={() => setAutoRotate((v) => !v)}
                title={autoRotate ? "Stop rotating" : "Auto-rotate"}
              >
                <Rotate3d size={16} />
              </button>
              {asset.params.rigged && (
                <>
                  <button
                    className={`icon-btn ${showBones ? "active" : ""}`}
                    onClick={() => setShowBones((v) => !v)}
                    title={showBones ? "Hide bones" : "Show bones"}
                  >
                    <Bone size={16} />
                  </button>
                  {asset.params.animated ? (
                    <button
                      className={`icon-btn ${playing ? "active" : ""}`}
                      onClick={() => setPlaying((v) => !v)}
                      title={playing ? "Pause the animation" : "Play the animation"}
                    >
                      {playing ? <Pause size={16} /> : <Play size={16} />}
                    </button>
                  ) : null}
                  {clipNames.length > 1 && (
                    <select
                      className="select clip-select"
                      value={clip ?? clipNames.length - 1}
                      onChange={(e) => setClip(Number(e.target.value))}
                      title="Animation to play"
                    >
                      {clipNames.map((name, i) => (
                        <option key={i} value={i}>
                          {name}
                        </option>
                      ))}
                    </select>
                  )}
                  {asset.params.animated ? (
                    <button
                      className={`icon-btn ${inPlace ? "active" : ""}`}
                      onClick={() => setInPlace((v) => !v)}
                      title={inPlace ? "In place: the root's travel is held (the GLB keeps it). Click to show the travel" : "Showing the travel: click to play in place"}
                    >
                      <Anchor size={16} />
                    </button>
                  ) : (
                    <button
                      className={`icon-btn ${posing ? "active" : ""}`}
                      onClick={() => setPosing((v) => !v)}
                      title={posing ? "Stop the pose test" : "Pose test: sway every bone to check the skin weights"}
                    >
                      <PersonStanding size={16} />
                    </button>
                  )}
                </>
              )}
            </div>
          </div>
        ) : asset.kind === "multiview" && asset.views ? (
          <div className="frame mv-frame">
            <div className="mv-grid">
              {asset.views.map((v) => (
                <a key={v.azimuth} href={v.url} target="_blank" rel="noreferrer" className="mv-view">
                  <img src={v.url} alt={`${v.azimuth}°`} />
                  <span className="mv-label">
                    {VIEW_NAMES[v.azimuth] ?? ""} {v.azimuth}°
                  </span>
                </a>
              ))}
            </div>
          </div>
        ) : (
          <a href={asset.url} target="_blank" rel="noreferrer" className="frame" style={{ aspectRatio: `${asset.width} / ${asset.height}` }}>
            <img src={asset.url} alt={asset.prompt} />
          </a>
        )}
      </div>
      <div className="toolbar">
        {(asset.reference_url || asset.input_url) && (
          <a href={(asset.reference_url || asset.input_url)!} target="_blank" rel="noreferrer" className="toolbar-ref" title="Reference used (open full size)">
            <img src={(asset.reference_url || asset.input_url)!} alt="Reference" />
          </a>
        )}
        <p className="toolbar-prompt" title={asset.prompt}>
          {asset.kind === "model3d"
            ? `${modelLabel(asset.params)} · ${asset.params.faces?.toLocaleString() ?? "?"} faces`
            : asset.prompt || (asset.kind === "multiview" ? "Multiview from image" : "")}
        </p>
        <div className="toolbar-actions">
          <button className={`tool-btn ${asset.favorite ? "fav" : ""}`} onClick={() => onToggleFavorite(asset)} title="Favorite">
            <Star size={15} fill={asset.favorite ? "currentColor" : "none"} />
          </button>
          {asset.prompt && (
            <button className="tool-btn" onClick={() => copy(asset.prompt, "Prompt")} title="Copy prompt">
              <Copy size={15} />
            </button>
          )}
          {asset.kind === "image" && (
            <button className="tool-btn" onClick={() => onMakeMultiview(asset)} title="Generate multi-view images of this">
              <Layers size={15} /> Multiview
            </button>
          )}
          {(asset.kind === "image" || asset.kind === "multiview") && (
            <button className="tool-btn" onClick={() => onMake3D(asset)} title="Reconstruct a 3D model from this">
              <Box size={15} /> 3D
            </button>
          )}
          {asset.kind === "model3d" && (
            <button className="tool-btn" onClick={() => onSegment(asset)} title="Split into parts (PartField)">
              <Puzzle size={15} /> Segment
            </button>
          )}
          {asset.kind === "model3d" && (
            <button className="tool-btn" onClick={() => onRemesh(asset)} title="Rebuild as a clean low-poly mesh (MeshFlow)">
              <Pyramid size={15} /> Remesh
            </button>
          )}
          {asset.kind === "model3d" && (
            <button className="tool-btn" onClick={() => onTexture(asset)} title="Generate a texture for this model">
              <PaintBucket size={15} /> Texture
            </button>
          )}
          {asset.kind === "model3d" && (
            <button className="tool-btn" onClick={() => onRig(asset)} title="Add a skeleton and skin weights (SkinTokens)">
              <Bone size={15} /> Rig
            </button>
          )}
          {asset.kind === "model3d" && asset.params.rigged && (
            <button className="tool-btn" onClick={() => onAnimate(asset)} title="Animate from a text prompt (UniMate)">
              <Footprints size={15} /> Animate
            </button>
          )}
          {asset.kind !== "model3d" && (
            <button className="tool-btn" onClick={() => onReuse(asset)} title="Load these settings into the panel">
              <RotateCcw size={15} /> Reuse
            </button>
          )}
          {asset.kind === "model3d" && asset.model_urls ? (
            <>
              <a className="tool-btn" href={asset.model_urls.glb} download title={`Download GLB (${asset.params.textured ? "textured" : "vertex colours"}${asset.params.rigged ? ", rigged" : ""}${asset.params.animated ? ", animated" : ""}${asset.params.segmented ? ", one mesh per part" : ""})`}>
                <Download size={15} /> GLB
              </a>
              {asset.model_urls.glb_full && (
                <a
                  className="tool-btn"
                  href={asset.model_urls.glb_full}
                  download
                  title={`Download the full mesh before the face limit (${asset.params.full_faces?.toLocaleString() ?? "?"} faces)`}
                >
                  <Download size={15} /> Full GLB
                </a>
              )}
              {asset.params.textured ? (
                <a className="tool-btn" href={`/api/assets/${asset.id}/obj.zip`} download title="Download OBJ + MTL + texture (zip)">
                  <Download size={15} /> OBJ
                </a>
              ) : (
                <a className="tool-btn" href={asset.model_urls.obj} download title="Download OBJ (vertex colours)">
                  <Download size={15} /> OBJ
                </a>
              )}
              {asset.image_urls?.texture && (
                <a className="tool-btn" href={asset.image_urls.texture} download title="Download the texture PNG">
                  <Download size={15} /> PNG
                </a>
              )}
            </>
          ) : (
            <a className="tool-btn" href={asset.url} download title={asset.kind === "multiview" ? "Download grid PNG" : "Download PNG"}>
              <Download size={15} />
            </a>
          )}
          {asset.kind !== "model3d" && (
            <a className="tool-btn" href={asset.url} target="_blank" rel="noreferrer" title="Open full size">
              <ExternalLink size={15} />
            </a>
          )}
          <button className="tool-btn danger" onClick={() => onDelete(asset)} title="Delete">
            <Trash2 size={15} />
          </button>
        </div>
      </div>
    </main>
  );
}
