import { ChevronDown, Pyramid, X, Zap } from "lucide-react";
import { useState } from "react";
import type { Asset } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export interface RemeshForm {
  count: number;
  fillHoles: boolean;
  steps: number;
  seed: number;
  randomSeed: boolean;
}

interface Props {
  form: RemeshForm;
  setForm: (f: Partial<RemeshForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  selected: Asset | null;
  available: boolean;
  onGenerate: () => void;
  submitting: boolean;
}

export function RemeshPanel(p: Props) {
  const { form, setForm, model } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const canGenerate = p.available && model !== null && !p.submitting;

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
              Rebuilds the shape as a clean, artist-style low-poly mesh (~0.5–2k faces) · colours carry over from the
              source as vertex colours · texture it afterwards for detail
            </span>
          </div>
        </Section>

        <Section label="Variations">
          <Segmented
            options={[1, 2, 4].map((n) => ({ value: n, label: n }))}
            value={form.count}
            onChange={(count) => setForm({ count })}
          />
          <p className="hint">Each seed gives a different topology; pick the cleanest.</p>
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

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Steps" min={8} max={60} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
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
