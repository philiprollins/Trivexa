import { ChevronDown, ImagePlus, Layers, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset, Checkpoint, MultiviewSupport, Sampler } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export interface MultiviewForm {
  mode: "t2mv" | "i2mv";
  prompt: string; // text-to-multiview
  imagePrompt: string; // optional description in image-to-multiview
  negative: string;
  model: string;
  steps: number;
  cfg: number;
  sampler: string;
  seed: number;
  randomSeed: boolean;
  count: number;
  removeBg: boolean;
  referenceScale: number;
}

export type Reference = { type: "asset"; asset: Asset } | { type: "upload"; dataUrl: string; name: string };

// Defaults from the MV-Adapter inference scripts.
export const MODE_CFG = { t2mv: 7, i2mv: 3 } as const;

interface Props {
  form: MultiviewForm;
  setForm: (f: Partial<MultiviewForm>) => void;
  reference: Reference | null;
  setReference: (r: Reference | null) => void;
  selectedImage: Asset | null;
  checkpoints: Checkpoint[];
  samplers: Sampler[];
  support: MultiviewSupport | null;
  onGenerate: () => void;
  submitting: boolean;
  notify: (msg: string, error?: boolean) => void;
}

const MAX_UPLOAD = 20 * 1024 * 1024;

export function MultiviewPanel(p: Props) {
  const { form, setForm } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const sdxl = p.checkpoints.filter((c) => c.arch === "sdxl");
  const adapterReady = p.support?.[form.mode] ?? false;
  const needsBg = form.mode === "i2mv" && form.removeBg && !p.support?.bg_removal;
  const hasInput = form.mode === "t2mv" ? form.prompt.trim().length > 0 : p.reference !== null;
  const canGenerate = adapterReady && hasInput && !needsBg && sdxl.some((c) => c.id === form.model) && !p.submitting;

  const readFile = (file: File | undefined) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) return p.notify("That file isn't an image", true);
    if (file.size > MAX_UPLOAD) return p.notify("Image is larger than 20 MB", true);
    const reader = new FileReader();
    reader.onload = () => p.setReference({ type: "upload", dataUrl: reader.result as string, name: file.name });
    reader.readAsDataURL(file);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate) {
      e.preventDefault();
      p.onGenerate();
    }
  };

  const refSrc = p.reference?.type === "asset" ? p.reference.asset.thumb_url : p.reference?.dataUrl;

  return (
    <aside className="panel left-panel" onKeyDown={onKeyDown}>
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Layers size={15} /> Generate Multiview
        </h2>

        <div className="section">
          <Segmented
            options={[
              { value: "i2mv", label: "Image → Views" },
              { value: "t2mv", label: "Text → Views" },
            ]}
            value={form.mode}
            onChange={(mode) => setForm({ mode, cfg: MODE_CFG[mode] })}
          />
        </div>

        {form.mode === "i2mv" && (
          <div
            className={`drop-zone ${dragging ? "dragging" : ""} ${refSrc ? "filled" : ""}`}
            onClick={() => !refSrc && fileInput.current?.click()}
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
            {refSrc ? (
              <>
                <img src={refSrc} alt="Reference" />
                <button className="drop-clear" onClick={() => p.setReference(null)} title="Remove reference">
                  <X size={14} />
                </button>
              </>
            ) : (
              <div className="drop-empty">
                <ImagePlus size={22} />
                <span>Drop an image or click to upload</span>
                <span className="hint">PNG, JPG, WEBP · up to 20 MB</span>
              </div>
            )}
          </div>
        )}
        {form.mode === "i2mv" && p.selectedImage && p.reference?.type !== "asset" && (
          <button className="link-btn use-selected" onClick={() => p.setReference({ type: "asset", asset: p.selectedImage! })}>
            Use the selected gallery image
          </button>
        )}
        {form.mode === "i2mv" && p.reference?.type === "asset" && p.selectedImage && p.selectedImage.id !== p.reference.asset.id && (
          <button className="link-btn use-selected" onClick={() => p.setReference({ type: "asset", asset: p.selectedImage! })}>
            Switch to the selected gallery image
          </button>
        )}

        <div className="prompt-card">
          {form.mode === "t2mv" ? (
            <textarea
              className="prompt"
              placeholder="Describe a single object or character…"
              value={form.prompt}
              onChange={(e) => setForm({ prompt: e.target.value })}
              rows={5}
            />
          ) : (
            <textarea
              className="prompt"
              placeholder="Optional: describe the subject"
              value={form.imagePrompt}
              onChange={(e) => setForm({ imagePrompt: e.target.value })}
              rows={2}
            />
          )}
          <textarea
            className="prompt negative"
            placeholder="What to avoid…"
            value={form.negative}
            onChange={(e) => setForm({ negative: e.target.value })}
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
          <p className="hint">6 views · 768 × 768 · front, 45°, right, back, left, 315°</p>
        </Section>

        {form.mode === "i2mv" && (
          <Section label="Background">
            <Segmented
              options={[
                { value: "remove", label: "Remove", disabled: !p.support?.bg_removal },
                { value: "keep", label: "Keep" },
              ]}
              value={form.removeBg ? "remove" : "keep"}
              onChange={(v) => setForm({ removeBg: v === "remove" })}
            />
            <p className="hint">Removal isolates the subject (BiRefNet). Keep it if your image already has a transparent background.</p>
          </Section>
        )}

        <Section label="Sets">
          <Segmented options={[1, 2, 3, 4].map((n) => ({ value: n, label: n }))} value={form.count} onChange={(count) => setForm({ count })} />
        </Section>

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Steps" min={10} max={80} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
            <Slider label="CFG scale" min={1} max={12} step={0.5} value={form.cfg} onChange={(cfg) => setForm({ cfg })} />
            {form.mode === "i2mv" && (
              <Slider
                label="Reference strength"
                min={0}
                max={2}
                step={0.05}
                value={form.referenceScale}
                onChange={(referenceScale) => setForm({ referenceScale })}
              />
            )}
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

        {!adapterReady && (
          <p className="notice">
            MV-Adapter weights for this mode aren't downloaded yet. Expected at{" "}
            <code>data/models/mv-adapter/mvadapter_{form.mode}_sdxl.safetensors</code>.
          </p>
        )}
        {needsBg && <p className="notice">Background removal needs <code>background_removal/birefnet.safetensors</code>.</p>}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Generate views{form.count > 1 ? ` ×${form.count}` : ""}
        </button>
      </div>
    </aside>
  );
}
