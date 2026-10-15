import { ChevronDown, Footprints, X, Zap } from "lucide-react";
import { useState } from "react";
import type { Asset } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export interface AnimateForm {
  prompt: string;
  count: number;
  guidance: number;
  seed: number;
  randomSeed: boolean;
}

interface Props {
  form: AnimateForm;
  setForm: (f: Partial<AnimateForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  selected: Asset | null;
  available: boolean;
  onGenerate: () => void;
  submitting: boolean;
}

// Short motions in the style of UniMate's captions (which all describe "an object" doing something).
const SUGGESTIONS = ["walks forward", "runs in a circle", "jumps up and down", "waves its right hand", "dances happily", "looks around"];

export function AnimatePanel(p: Props) {
  const { form, setForm, model } = p;
  const [showAdvanced, setShowAdvanced] = useState(false);
  const rigged = !!model?.params.rigged;
  const canGenerate = p.available && rigged && form.prompt.trim().length > 0 && !p.submitting;

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate) {
      e.preventDefault();
      p.onGenerate();
    }
  };

  return (
    <aside className="panel left-panel" onKeyDown={onKeyDown}>
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Footprints size={15} /> Animate Rigged Model
        </h2>

        <Section label="Model to animate">
          {model ? (
            <div className="source-card">
              <img src={model.thumb_url} alt="" />
              <div>
                <strong>{model.params.animated ? "Animated model" : "Rigged model"}</strong>
                <span>{model.params.bones ?? "?"} bones</span>
              </div>
              <button className="drop-clear static" onClick={() => p.setModel(null)} title="Clear">
                <X size={14} />
              </button>
            </div>
          ) : (
            <p className="hint">Select a rigged model in the gallery, or press Animate on one in the viewer.</p>
          )}
          {model && !rigged && <p className="notice">This model has no skeleton yet: rig it first.</p>}
          {model?.params.animated && (
            <p className="hint">
              The new motion is added to this model's {model.params.clips?.length ?? "existing"} animation
              {model.params.clips?.length === 1 ? "" : "s"}; animate the rigged model it came from for a fresh one.
            </p>
          )}
          {p.selected?.kind === "model3d" && p.selected.params.rigged && p.selected.id !== model?.id && (
            <button className="link-btn use-selected" onClick={() => p.setModel(p.selected)}>
              Use the selected rigged model
            </button>
          )}
        </Section>

        <div className="prompt-card">
          <textarea
            className="prompt"
            placeholder="Describe the motion, e.g. “walks forward, then turns left”…"
            value={form.prompt}
            onChange={(e) => setForm({ prompt: e.target.value })}
            rows={3}
          />
          <div className="prompt-footer">
            <span className="hint">Read as “An object …”</span>
            <span className="hint">Ctrl + Enter</span>
          </div>
        </div>
        <div className="suggestions">
          {SUGGESTIONS.map((s) => (
            <button key={s} className="suggestion" onClick={() => setForm({ prompt: s })}>
              {s}
            </button>
          ))}
        </div>

        <Section label="Engine">
          <div className="engine-card">
            <strong>UniMate</strong>
            <span>
              Text-to-motion for any skeleton: the joints are labelled from the rig's shape, then 2 s of motion at 30 fps
              drives the bones · ~15 s · weights are non-commercial (CC BY-NC)
            </span>
          </div>
        </Section>

        <Section label="Variations">
          <Segmented
            options={[1, 2, 4].map((n) => ({ value: n, label: n }))}
            value={form.count}
            onChange={(count) => setForm({ count })}
          />
          <p className="hint">Each seed gives a different take on the motion; contact-heavy ones can slide or float.</p>
        </Section>

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Guidance" min={1} max={8} step={0.5} value={form.guidance} onChange={(guidance) => setForm({ guidance })} />
            <p className="hint">How closely the motion follows the text (UniMate's default is 3); 1 ignores it.</p>
            <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
          </div>
        )}

        {!p.available && (
          <p className="notice">
            UniMate weights not found. Run <code>uv run python -m trivexa.download unimate</code> in <code>backend</code>.
          </p>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Animate{form.count > 1 ? ` ×${form.count}` : ""}
        </button>
      </div>
    </aside>
  );
}
