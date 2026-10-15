import { Puzzle, X, Zap } from "lucide-react";
import type { Asset } from "../api";
import { Section, Segmented, Slider } from "./controls";

export type SegmentDetail = "simple" | "balanced" | "detailed" | "custom";

export interface SegmentForm {
  detail: SegmentDetail;
  parts: number; // used when detail is "custom"
}

// Tripo's presets (Simple 3-6, Balanced 6-15, Detailed 15+ parts); the backend's partfield.DETAIL_PARTS.
export const DETAIL_PARTS: Record<Exclude<SegmentDetail, "custom">, number> = { simple: 5, balanced: 10, detailed: 18 };
export const MAX_PARTS = 32;

export const segmentParts = (f: SegmentForm) => (f.detail === "custom" ? f.parts : DETAIL_PARTS[f.detail]);

interface Props {
  form: SegmentForm;
  setForm: (f: Partial<SegmentForm>) => void;
  model: Asset | null;
  setModel: (a: Asset | null) => void;
  selected: Asset | null;
  available: boolean;
  onGenerate: () => void;
  submitting: boolean;
}

const DETAIL_HINTS: Record<SegmentDetail, string> = {
  simple: "About 5 parts: the main masses, e.g. head, body and limbs.",
  balanced: "About 10 parts: limbs split at the joints, separate accessories.",
  detailed: "About 18 parts: hands, feet and small details get their own parts.",
  custom: "Exactly this many parts. Every part is one connected region of the surface.",
};

const modelLabel = (a: Asset) =>
  a.params.segmented
    ? `Segmented model · ${a.params.parts} parts`
    : a.params.rigged
    ? "Rigged model"
    : a.params.textured
    ? "Textured model"
    : a.params.remeshed
    ? "Remeshed model"
    : "3D model";

export function SegmentPanel(p: Props) {
  const { form, setForm, model } = p;
  const canGenerate = p.available && model !== null && !p.submitting;

  return (
    <aside className="panel left-panel">
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Puzzle size={15} /> Segment 3D Model
        </h2>

        <Section label="Model to segment">
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
            <p className="hint">Select a 3D model in the gallery, or press Segment on one in the viewer.</p>
          )}
          {p.selected?.kind === "model3d" && p.selected.id !== model?.id && (
            <button className="link-btn use-selected" onClick={() => p.setModel(p.selected)}>
              Use the selected 3D model
            </button>
          )}
          {model?.params.rigged && <p className="hint">The parts are plain meshes: the rig is not kept.</p>}
        </Section>

        <Section label="Engine">
          <div className="engine-card">
            <strong>PartField</strong>
            <span>
              Learned part features, clustered over the surface · each part is a separate mesh in the GLB and OBJ, keeping
              the model's colours or texture · ~10 s for 200k faces
            </span>
          </div>
        </Section>

        <Section label="Detail level">
          <Segmented
            options={[
              { value: "simple", label: "Simple" },
              { value: "balanced", label: "Balanced" },
              { value: "detailed", label: "Detailed" },
              { value: "custom", label: "Custom" },
            ]}
            value={form.detail}
            onChange={(detail) => setForm(detail === "custom" ? { detail, parts: segmentParts(form) } : { detail })}
          />
          <p className="hint">{DETAIL_HINTS[form.detail]}</p>
        </Section>

        {form.detail === "custom" && (
          <Slider label="Parts" min={2} max={MAX_PARTS} step={1} value={form.parts} onChange={(parts) => setForm({ parts })} />
        )}

        {!p.available && (
          <p className="notice">
            PartField weights not found. Run <code>uv run python -m trivexa.download partfield</code> in <code>backend</code>.
          </p>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={p.onGenerate}>
          <Zap size={16} fill="currentColor" />
          Segment into {segmentParts(form)} parts
        </button>
      </div>
    </aside>
  );
}
