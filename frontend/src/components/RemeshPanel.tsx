import { ChevronDown, ImagePlus, Pyramid, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset, RemeshDetail } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";
import type { TextureReference } from "./TexturePanel";

export interface RemeshForm {
  count: number;
  fillHoles: boolean;
  detail: RemeshDetail;
  steps: number;
  seed: number;
  randomSeed: boolean;
  useImage: boolean;
  removeBg: boolean;
  guidance: number;
}

interface Props {
  form: RemeshForm;
  setForm: (f: Partial<RemeshForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  reference: TextureReference | null; // null = the model's own input image
  setReference: (r: TextureReference | null) => void;
  selected: Asset | null;
  available: boolean;
  imageAvailable: boolean; // DINOv3 weights present
  bgAvailable: boolean;
  onGenerate: () => void;
  submitting: boolean;
  notify: (msg: string, error?: boolean) => void;
}

const MAX_UPLOAD = 20 * 1024 * 1024;

// Relative face counts and image-guided times measured on a robot, a teapot and a chair.
const DETAIL_HINTS: Record<RemeshDetail, string> = {
  low: "~0.8× the faces, ~7 s.",
  normal: "What MeshFlow was trained for, ~10 s.",
  high: "~1.6× the faces on curved shapes, ~17 s.",
  max: "Experimental: 3–6× the faces on curved shapes but more overlapping faces, ~25 s.",
};

export function RemeshPanel(p: Props) {
  const { form, setForm, model } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const useImage = form.useImage && p.imageAvailable;
  const canGenerate = p.available && model !== null && !(useImage && form.removeBg && !p.bgAvailable) && !p.submitting;

  const readFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return p.notify("That file isn't an image", true);
    if (file.size > MAX_UPLOAD) return p.notify("Image is larger than 20 MB", true);
    const reader = new FileReader();
    reader.onload = () => p.setReference({ type: "upload", dataUrl: reader.result as string, name: file.name });
    reader.readAsDataURL(file);
  };

  const refSrc = p.reference
    ? p.reference.type === "asset"
      ? p.reference.asset.thumb_url
      : p.reference.dataUrl
    : model?.input_url;

  return (
    <aside className="panel left-panel">
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Pyramid size={15} /> Remesh 3D Model
        </h2>

        <Section label="Model to remesh">
          {model ? (
            <div className="source-card">
              <img src={model.thumb_url} alt="" />
              <div>
                <strong>{model.params.remeshed ? "Remeshed model" : model.params.textured ? "Textured model" : "3D model"}</strong>
                <span>{model.params.faces?.toLocaleString()} faces</span>
              </div>
              <button className="drop-clear static" onClick={() => p.setModel(null)} title="Clear">
                <X size={14} />
              </button>
            </div>
          ) : (
            <p className="hint">Select a 3D model in the gallery, or press Remesh on one in the viewer.</p>
          )}
          {p.selected?.kind === "model3d" && p.selected.id !== model?.id && (
            <button className="link-btn use-selected" onClick={() => p.setModel(p.selected)}>
              Use the selected 3D model
            </button>
          )}
        </Section>

        <Section label="Engine">
          <div className="engine-card">
            <strong>MeshFlow</strong>
            <span>
              Rebuilds the shape as a clean, artist-style low-poly mesh (~1–2k faces at normal detail) · colours carry over from the
              source as vertex colours · texture it afterwards for detail
            </span>
          </div>
        </Section>

        <Section label="Guide with">
          <Segmented
            options={[
              { value: "image", label: "Shape + image", disabled: !p.imageAvailable },
              { value: "shape", label: "Shape only" },
            ]}
            value={useImage ? "image" : "shape"}
            onChange={(v) => setForm({ useImage: v === "image" })}
          />
          <p className="hint">
            {p.imageAvailable
              ? "An image keeps thin parts and small details (antennae, fingers) that shape-only smooths away; ~2× slower."
              : "Image guidance needs DINOv3 ViT-L/16 in data/models/meshflow."}
          </p>
        </Section>

        {useImage && (
          <Section label="Reference image">
            <div className="ref-row">
              <div className="ref-thumb">{refSrc ? <img src={refSrc} alt="Reference" /> : <ImagePlus size={18} />}</div>
              <div className="ref-actions">
                <span className="hint">{p.reference ? "Custom reference" : "The model's own input image"}</span>
                <div className="ref-buttons">
                  <button className="link-btn" onClick={() => fileInput.current?.click()}>
                    Upload…
                  </button>
                  {p.selected?.kind === "image" && (
                    <button className="link-btn" onClick={() => p.setReference({ type: "asset", asset: p.selected! })}>
                      Use selected image
                    </button>
                  )}
                  {p.reference && (
                    <button className="link-btn" onClick={() => p.setReference(null)}>
                      Reset
                    </button>
                  )}
                </div>
              </div>
              <input ref={fileInput} type="file" accept="image/*" hidden onChange={(e) => readFile(e.target.files?.[0])} />
            </div>
          </Section>
        )}

        <Section label="Detail">
          <Segmented
            options={[
              { value: "low", label: "Low" },
              { value: "normal", label: "Normal" },
              { value: "high", label: "High" },
              { value: "max", label: "Max" },
            ]}
            value={form.detail}
            onChange={(detail) => setForm({ detail })}
          />
          <p className="hint">
            {DETAIL_HINTS[form.detail]} Sets how dense the mesh may get, not an exact face count; flat, hard-surface
            parts stay simple.
          </p>
        </Section>

        <Section label="Variations">
          <Segmented
            options={[1, 2, 4].map((n) => ({ value: n, label: n }))}
            value={form.count}
            onChange={(count) => setForm({ count })}
          />
          <p className="hint">Each seed gives a different topology and face count; pick the cleanest.</p>
        </Section>

        <Section label="Small holes">
          <Segmented
            options={[
              { value: "keep", label: "Leave open" },
              { value: "fill", label: "Fill" },
            ]}
            value={form.fillHoles ? "fill" : "keep"}
            onChange={(v) => setForm({ fillHoles: v === "fill" })}
          />
        </Section>

        {useImage && (
          <Section label="Reference background">
            <Segmented
              options={[
                { value: "remove", label: "Remove", disabled: !p.bgAvailable },
                { value: "keep", label: "Keep" },
              ]}
              value={form.removeBg ? "remove" : "keep"}
              onChange={(v) => setForm({ removeBg: v === "remove" })}
            />
          </Section>
        )}

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Steps" min={8} max={60} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
            {useImage && (
              <Slider
                label="Image guidance"
                min={1}
                max={6}
                step={0.25}
                value={form.guidance}
                onChange={(guidance) => setForm({ guidance })}
              />
            )}
            <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
          </div>
        )}

        {!p.available && (
          <p className="notice">
            MeshFlow weights not found. Expected <code>config.yaml</code> and <code>model.pth</code> in{" "}
            <code>data/models/meshflow</code>.
          </p>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Remesh{form.count > 1 ? ` ×${form.count}` : ""}
        </button>
      </div>
    </aside>
  );
}
