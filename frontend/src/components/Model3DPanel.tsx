import { Box, ImagePlus, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset } from "../api";
import { Section, Segmented, Slider } from "./controls";

export interface Model3DForm {
  removeBg: boolean;
  foregroundRatio: number;
  resolution: number;
}

export type Source3D =
  | { type: "asset"; asset: Asset; view: number }
  | { type: "upload"; dataUrl: string; name: string };

interface Props {
  form: Model3DForm;
  setForm: (f: Partial<Model3DForm>) => void;
  source: Source3D | null;
  setSource: (s: Source3D | null) => void;
  selected: Asset | null; // image or multiview asset currently selected in the gallery
  available: boolean;
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
  const fileInput = useRef<HTMLInputElement>(null);
  const canGenerate = p.available && source !== null && !(form.removeBg && !p.bgAvailable) && !p.submitting;

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
            <p className="hint">TripoSR reconstructs from a single image; the front view usually works best.</p>
          </Section>
        )}

        <Section label="Engine">
          <div className="engine-card">
            <strong>TripoSR</strong>
            <span>Fast single-image reconstruction · vertex colours · good for prototypes and small props</span>
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

        {!p.available && (
          <p className="notice">
            TripoSR weights not found. Expected <code>config.yaml</code>, <code>model.ckpt</code> and{" "}
            <code>dino/config.json</code> in <code>models/triposr</code>.
          </p>
        )}
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
