import { Box, ChevronDown, ImagePlus, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export type Engine3D = "triposr" | "triposg";

export interface Model3DForm {
  engine: Engine3D;
  removeBg: boolean;
  // TripoSR
  foregroundRatio: number;
  resolution: number;
  // TripoSG
  octreeDepth: 8 | 9;
  faceLimit: number; // 0 = keep every face
  steps: number;
  guidance: number;
  seed: number;
  randomSeed: boolean;
}

const FACE_LIMITS = [
  { value: 50_000, label: "50k" },
  { value: 200_000, label: "200k" },
  { value: 500_000, label: "500k" },
  { value: 0, label: "Full" },
];

export type Source3D =
  | { type: "asset"; asset: Asset; view: number }
  | { type: "upload"; dataUrl: string; name: string };

interface Props {
  form: Model3DForm;
  setForm: (f: Partial<Model3DForm>) => void;
  source: Source3D | null;
  setSource: (s: Source3D | null) => void;
  selected: Asset | null; // image or multiview asset currently selected in the gallery
  available: Record<Engine3D, boolean>;
  bgAvailable: boolean;
  onGenerate: () => void;
  submitting: boolean;
  notify: (msg: string, error?: boolean) => void;
}

const MAX_UPLOAD = 20 * 1024 * 1024;

export function sourcePreview(s: Source3D): string {
  if (s.type === "upload") return s.dataUrl;
  if (s.asset.kind === "multiview") return s.asset.views?.find((v) => v.azimuth === s.view)?.url ?? s.asset.thumb_url;
  return s.asset.thumb_url;
}

export function Model3DPanel(p: Props) {
  const { form, setForm, source } = p;
  const [dragging, setDragging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const sg = form.engine === "triposg";
  const canGenerate = p.available[form.engine] && source !== null && !(form.removeBg && !p.bgAvailable) && !p.submitting;

  const readFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return p.notify("That file isn't an image", true);
    if (file.size > MAX_UPLOAD) return p.notify("Image is larger than 20 MB", true);
    const reader = new FileReader();
    reader.onload = () => p.setSource({ type: "upload", dataUrl: reader.result as string, name: file.name });
    reader.readAsDataURL(file);
  };

  const selectedIsNew =
    p.selected && (p.selected.kind === "image" || p.selected.kind === "multiview") &&
    !(source?.type === "asset" && source.asset.id === p.selected.id);

  return (
    <aside className="panel left-panel">
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Box size={15} /> Generate 3D Model
        </h2>

        <div
          className={`drop-zone ${dragging ? "dragging" : ""} ${source ? "filled" : ""}`}
          onClick={() => !source && fileInput.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            readFile(e.dataTransfer.files[0]);
          }}
        >
          <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => readFile(e.target.files?.[0])} />
          {source ? (
            <>
              <img src={sourcePreview(source)} alt="Input" />
              <button className="drop-clear" onClick={() => p.setSource(null)} title="Remove input">
                <X size={14} />
              </button>
            </>
          ) : (
            <div className="drop-empty">
              <ImagePlus size={22} />
              <span>Drop an image or click to upload</span>
              <span className="hint">One subject, ideally on a plain background</span>
            </div>
          )}
        </div>
        {selectedIsNew && (
          <button
            className="link-btn use-selected"
            onClick={() => p.setSource({ type: "asset", asset: p.selected!, view: 0 })}
          >
            Use the selected gallery {p.selected!.kind === "multiview" ? "multiview set" : "image"}
          </button>
        )}

        {source?.type === "asset" && source.asset.kind === "multiview" && source.asset.views && (
          <Section label="View to reconstruct from">
            <select
              className="select"
              value={source.view}
              onChange={(e) => p.setSource({ ...source, view: Number(e.target.value) })}
            >
              {source.asset.views.map((v) => (
                <option key={v.azimuth} value={v.azimuth}>
                  {v.azimuth === 0 ? "Front 0°" : `${v.azimuth}°`}
                </option>
              ))}
            </select>
            <p className="hint">Both engines reconstruct from a single image; the front view usually works best.</p>
          </Section>
        )}

        <Section label="Engine">
          <Segmented
            options={[
              { value: "triposg", label: "TripoSG" },
              { value: "triposr", label: "TripoSR" },
            ]}
            value={form.engine}
            onChange={(engine) => setForm({ engine })}
          />
          <div className="engine-card">
            {sg ? (
              <span>
                High-fidelity shape from a 1.5B flow model · sharp detail and clean surfaces · no colour (texture it
                next) · ~30–60 s
              </span>
            ) : (
              <span>Fast single-image reconstruction · vertex colours · soft shapes · good for quick prototypes</span>
            )}
          </div>
        </Section>

        <Section label="Background">
          <Segmented
            options={[
              { value: "remove", label: "Remove", disabled: !p.bgAvailable },
              { value: "keep", label: "Keep" },
            ]}
            value={form.removeBg ? "remove" : "keep"}
            onChange={(v) => setForm({ removeBg: v === "remove" })}
          />
        </Section>

        {sg ? (
          <>
            <Section label="Mesh resolution">
              <Segmented
                options={[
                  { value: 8, label: "256" },
                  { value: 9, label: "512" },
                ]}
                value={form.octreeDepth}
                onChange={(octreeDepth) => setForm({ octreeDepth })}
              />
              <p className="hint">Surface grid size. 512 keeps fine detail; 256 is quicker and softer.</p>
            </Section>

            <Section label="Face limit">
              <Segmented options={FACE_LIMITS} value={form.faceLimit} onChange={(faceLimit) => setForm({ faceLimit })} />
              <p className="hint">Decimates the raw mesh (often over a million faces) to keep files and later steps light.</p>
            </Section>

            <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
              Sampling settings
              <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
            </button>
            {showAdvanced && (
              <div className="advanced">
                <Slider label="Steps" min={10} max={100} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
                <Slider
                  label="Guidance"
                  min={1}
                  max={12}
                  step={0.5}
                  value={form.guidance}
                  onChange={(guidance) => setForm({ guidance })}
                />
                <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
              </div>
            )}
          </>
        ) : (
          <>
            <Slider
              label="Subject fill"
              min={0.5}
              max={1}
              step={0.05}
              value={form.foregroundRatio}
              onChange={(foregroundRatio) => setForm({ foregroundRatio })}
            />

            <Section label="Mesh resolution">
              <Segmented
                options={[192, 256, 320, 384].map((n) => ({ value: n, label: n }))}
                value={form.resolution}
                onChange={(resolution) => setForm({ resolution })}
              />
              <p className="hint">Marching-cubes grid size. Higher gives finer surfaces and larger files.</p>
            </Section>
          </>
        )}

        {!p.available[form.engine] &&
          (sg ? (
            <p className="notice">
              TripoSG weights not found in <code>data/models/triposg</code>. Run{" "}
              <code>uv run python -m trivexa.download triposg</code>.
            </p>
          ) : (
            <p className="notice">
              TripoSR weights not found. Expected <code>config.yaml</code>, <code>model.ckpt</code> and{" "}
              <code>dino/config.json</code> in <code>models/triposr</code>.
            </p>
          ))}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Generate 3D model
        </button>
      </div>
    </aside>
  );
}
