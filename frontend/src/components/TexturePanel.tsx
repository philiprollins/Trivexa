import { ChevronDown, ImagePlus, PaintBucket, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset, Checkpoint, Sampler, TextureBaker } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export interface TextureForm {
  prompt: string;
  model: string;
  removeBg: boolean;
  steps: number;
  cfg: number;
  sampler: string;
  seed: number;
  randomSeed: boolean;
  referenceScale: number;
  uvSize: number;
}

export type TextureReference = { type: "asset"; asset: Asset } | { type: "upload"; dataUrl: string; name: string };

/** Older TripoSR models (before facing was recorded) face -Z, i.e. need a half turn. */
export const defaultFrontYaw = (model: Asset) => (model.params.front === "+z" ? 0 : model.params.engine === "triposr" ? 180 : 0);

interface Props {
  form: TextureForm;
  setForm: (f: Partial<TextureForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  frontYaw: number;
  setFrontYaw: (deg: number) => void;
  reference: TextureReference | null; // null = the model's own input image
  setReference: (r: TextureReference | null) => void;
  selected: Asset | null;
  checkpoints: Checkpoint[];
  samplers: Sampler[];
  available: boolean;
  bgAvailable: boolean;
  baker: TextureBaker | null;
  onGenerate: () => void;
  submitting: boolean;
  notify: (msg: string, error?: boolean) => void;
}

const MAX_UPLOAD = 20 * 1024 * 1024;

export function TexturePanel(p: Props) {
  const { form, setForm, model } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const sdxl = p.checkpoints.filter((c) => c.arch === "sdxl");
  const canGenerate =
    p.available && model !== null && sdxl.some((c) => c.id === form.model) && !(form.removeBg && !p.bgAvailable) && !p.submitting;

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
          <PaintBucket size={15} /> Texture 3D Model
        </h2>

        <Section label="Model to texture">
          {model ? (
            <div className="source-card">
              <img src={model.thumb_url} alt="" />
              <div>
                <strong>{model.params.textured ? "Textured model" : "3D model"}</strong>
                <span>{model.params.faces?.toLocaleString()} faces</span>
              </div>
              <button className="drop-clear static" onClick={() => p.setModel(null)} title="Clear">
                <X size={14} />
              </button>
            </div>
          ) : (
            <p className="hint">Select a 3D model in the gallery, or press Texture on one in the viewer.</p>
          )}
          {p.selected?.kind === "model3d" && p.selected.id !== model?.id && (
            <button className="link-btn use-selected" onClick={() => p.setModel(p.selected)}>
              Use the selected 3D model
            </button>
          )}
        </Section>

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

        <Section label="Front of the model faces">
          <Segmented
            options={[
              { value: 0, label: "Front" },
              { value: 90, label: "Right" },
              { value: 180, label: "Back" },
              { value: 270, label: "Left" },
            ]}
            value={p.frontYaw}
            onChange={p.setFrontYaw}
          />
          <p className="hint">Turns the model before texturing so the reference lines up with its front.</p>
        </Section>

        <div className="prompt-card">
          <textarea
            className="prompt"
            placeholder="Optional: describe the materials"
            value={form.prompt}
            onChange={(e) => setForm({ prompt: e.target.value })}
            rows={2}
          />
        </div>

        <Section label="Base model (SDXL)">
          <select className="select" value={form.model} onChange={(e) => setForm({ model: e.target.value })}>
            {sdxl.length === 0 && <option value="">No SDXL checkpoints found</option>}
            {sdxl.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </Section>

        {p.baker && (
          <Section label="Baker">
            <div className="engine-card">
              <strong>{p.baker.baker === "upstream" ? "MV-Adapter (upstream)" : "Lean (Trivexa)"}</strong>
              <span>
                {p.baker.baker === "upstream"
                  ? "UVAtlas unwrap, RealESRGAN ×2 views, LaMa inpainting of unseen areas"
                  : "xatlas unwrap, OpenGL projection, vertex colours where no view sees"}
              </span>
              {p.baker.note && <span className="warn-text">Fallback: {p.baker.note}</span>}
              <span>Change in settings.toml ([texture] baker).</span>
            </div>
          </Section>
        )}

        <Section label="Texture size">
          <Segmented
            options={[1024, 2048, 4096].map((n) => ({ value: n, label: `${n / 1024}K` }))}
            value={form.uvSize}
            onChange={(uvSize) => setForm({ uvSize })}
          />
        </Section>

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

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Steps" min={10} max={80} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
            <Slider label="CFG scale" min={1} max={12} step={0.5} value={form.cfg} onChange={(cfg) => setForm({ cfg })} />
            <Slider
              label="Reference strength"
              min={0}
              max={2}
              step={0.05}
              value={form.referenceScale}
              onChange={(referenceScale) => setForm({ referenceScale })}
            />
            <Section label="Sampler">
              <select className="select" value={form.sampler} onChange={(e) => setForm({ sampler: e.target.value })}>
                {p.samplers.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </Section>
            <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
          </div>
        )}

        {!p.available && (
          <p className="notice">
            MV-Adapter ig2mv weights aren't downloaded yet. Expected at{" "}
            <code>data/models/mv-adapter/mvadapter_ig2mv_sdxl.safetensors</code>.
          </p>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Generate texture
        </button>
      </div>
    </aside>
  );
}
