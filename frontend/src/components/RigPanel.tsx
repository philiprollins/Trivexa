import { Bone, ChevronDown, X, Zap } from "lucide-react";
import { useState } from "react";
import type { Asset, RigStyle } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export interface RigForm {
  style: RigStyle;
  count: number;
  beams: number;
  temperature: number;
  seed: number;
  randomSeed: boolean;
}

interface Props {
  form: RigForm;
  setForm: (f: Partial<RigForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  selected: Asset | null;
  available: boolean;
  onGenerate: () => void;
  submitting: boolean;
}

const STYLE_HINTS: Record<RigStyle, string> = {
  auto: "Any kind of object: creatures, characters, props with moving parts. Bone count and layout follow the shape.",
  humanoid:
    "Asks for a skeleton like the VRoid humanoids SkinTokens learned from: spine, limbs and finger chains, about twice the bones of Auto on a character. Works best on a character standing in a T- or A-pose.",
};

const modelLabel = (a: Asset) =>
  a.params.rigged ? "Rigged model" : a.params.textured ? "Textured model" : a.params.remeshed ? "Remeshed model" : "3D model";

export function RigPanel(p: Props) {
  const { form, setForm, model } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const canGenerate = p.available && model !== null && !p.submitting;

  return (
    <aside className="panel left-panel">
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Bone size={15} /> Rig 3D Model
        </h2>

        <Section label="Model to rig">
          {model ? (
            <div className="source-card">
              <img src={model.thumb_url} alt="" />
              <div>
                <strong>{modelLabel(model)}</strong>
                <span>{model.params.faces?.toLocaleString()} faces</span>
              </div>
              <button className="drop-clear static" onClick={() => p.setModel(null)} title="Clear">
                <X size={14} />
              </button>
            </div>
          ) : (
            <p className="hint">Select a 3D model in the gallery, or press Rig on one in the viewer.</p>
          )}
          {p.selected?.kind === "model3d" && p.selected.id !== model?.id && (
            <button className="link-btn use-selected" onClick={() => p.setModel(p.selected)}>
              Use the selected 3D model
            </button>
          )}
        </Section>

        <Section label="Engine">
          <div className="engine-card">
            <strong>SkinTokens</strong>
            <span>
              Predicts a skeleton and per-vertex skin weights in one pass · the GLB keeps the model's colours or texture ·
              ~10–20 s
            </span>
          </div>
        </Section>

        <Section label="Skeleton">
          <Segmented
            options={[
              { value: "auto", label: "Auto" },
              { value: "humanoid", label: "Humanoid" },
            ]}
            value={form.style}
            onChange={(style) => setForm({ style })}
          />
          <p className="hint">{STYLE_HINTS[form.style]}</p>
        </Section>

        <Section label="Variations">
          <Segmented
            options={[1, 2, 4].map((n) => ({ value: n, label: n }))}
            value={form.count}
            onChange={(count) => setForm({ count })}
          />
          <p className="hint">Each seed gives a different skeleton; check them with the viewer's bones and pose test.</p>
        </Section>

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Beams" min={1} max={20} step={1} value={form.beams} onChange={(beams) => setForm({ beams })} />
            <p className="hint">More beams search for a more likely skeleton; fewer are faster and vary more.</p>
            <Slider
              label="Temperature"
              min={0.1}
              max={2}
              step={0.1}
              value={form.temperature}
              onChange={(temperature) => setForm({ temperature })}
            />
            <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
          </div>
        )}

        {!p.available && (
          <p className="notice">
            SkinTokens weights not found. Run <code>uv run python -m trivexa.download skintokens</code> in{" "}
            <code>backend</code>.
          </p>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Rig{form.count > 1 ? ` ×${form.count}` : ""}
        </button>
      </div>
    </aside>
  );
}
