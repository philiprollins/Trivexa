import { ChevronDown, Dices, Lock, Sparkles, Zap } from "lucide-react";
import { useState } from "react";
import type { Checkpoint, ImageParams, Sampler } from "../api";

export type ImageForm = Omit<ImageParams, "model_name" | "sampler_label"> & { count: number; randomSeed: boolean };

export const ASPECTS = [
  { label: "1:1", w: 1024, h: 1024 },
  { label: "3:4", w: 896, h: 1152 },
  { label: "2:3", w: 832, h: 1216 },
  { label: "9:16", w: 768, h: 1344 },
  { label: "4:3", w: 1152, h: 896 },
  { label: "3:2", w: 1216, h: 832 },
  { label: "16:9", w: 1344, h: 768 },
];

interface Props {
  form: ImageForm;
  setForm: (f: Partial<ImageForm>) => void;
  checkpoints: Checkpoint[];
  samplers: Sampler[];
  onGenerate: () => void;
  submitting: boolean;
}

export function ImagePanel({ form, setForm, checkpoints, samplers, onGenerate, submitting }: Props) {
  const [showNegative, setShowNegative] = useState(Boolean(form.negative));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const sdxl = checkpoints.filter((c) => c.arch === "sdxl");
  const otherCount = checkpoints.filter((c) => c.arch === "sd15").length;
  const canGenerate = form.prompt.trim().length > 0 && sdxl.some((c) => c.id === form.model) && !submitting;

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && canGenerate) {
      e.preventDefault();
      onGenerate();
    }
  };

  return (
    <aside className="panel left-panel" onKeyDown={onKeyDown}>
      <div className="panel-scroll">
        <h2 className="panel-title">
          <Sparkles size={15} /> Generate Image
        </h2>

        <div className="prompt-card">
          <textarea
            className="prompt"
            placeholder="Describe the image you want to create…"
            value={form.prompt}
            onChange={(e) => setForm({ prompt: e.target.value })}
            rows={6}
            autoFocus
          />
          <div className="prompt-footer">
            <button className="link-btn" onClick={() => setShowNegative((v) => !v)}>
              {showNegative ? "− Negative prompt" : "+ Negative prompt"}
            </button>
            <span className="hint">Ctrl + Enter</span>
          </div>
          {showNegative && (
            <textarea
              className="prompt negative"
              placeholder="What to avoid…"
              value={form.negative}
              onChange={(e) => setForm({ negative: e.target.value })}
              rows={3}
            />
          )}
        </div>

        <Section label="Model">
          <select className="select" value={form.model} onChange={(e) => setForm({ model: e.target.value })}>
            {sdxl.length === 0 && <option value="">No SDXL checkpoints found</option>}
            {sdxl.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
          {otherCount > 0 && <p className="hint">{otherCount} SD 1.5 checkpoints found — not supported yet.</p>}
        </Section>

        <Section label="Aspect ratio">
          <div className="aspect-grid">
            {ASPECTS.map((a) => {
              const active = a.w === form.width && a.h === form.height;
              const scale = 18 / Math.max(a.w, a.h);
              return (
                <button
                  key={a.label}
                  className={`aspect ${active ? "active" : ""}`}
                  onClick={() => setForm({ width: a.w, height: a.h })}
                  title={`${a.w} × ${a.h}`}
                >
                  <span className="aspect-box" style={{ width: a.w * scale, height: a.h * scale }} />
                  {a.label}
                </button>
              );
            })}
          </div>
          <p className="hint">
            {form.width} × {form.height}
          </p>
        </Section>

        <Section label="Images">
          <div className="segmented">
            {[1, 2, 3, 4].map((n) => (
              <button key={n} className={form.count === n ? "active" : ""} onClick={() => setForm({ count: n })}>
                {n}
              </button>
            ))}
          </div>
        </Section>

        <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
          Sampling settings
          <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
        </button>
        {showAdvanced && (
          <div className="advanced">
            <Slider label="Steps" min={1} max={80} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
            <Slider label="CFG scale" min={1} max={15} step={0.5} value={form.cfg} onChange={(cfg) => setForm({ cfg })} />
            <Section label="Sampler">
              <select className="select" value={form.sampler} onChange={(e) => setForm({ sampler: e.target.value })}>
                {samplers.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </Section>
            <Section label="Seed">
              <div className="seed-row">
                <input
                  className="input"
                  type="number"
                  min={0}
                  max={4294967295}
                  disabled={form.randomSeed}
                  value={form.randomSeed ? "" : form.seed}
                  placeholder="Random"
                  onChange={(e) => setForm({ seed: Math.max(0, Math.min(4294967295, Number(e.target.value) || 0)) })}
                />
                <button
                  className={`icon-btn ${form.randomSeed ? "active" : ""}`}
                  title={form.randomSeed ? "Random seed each time (click to lock)" : "Seed locked (click for random)"}
                  onClick={() => setForm({ randomSeed: !form.randomSeed })}
                >
                  {form.randomSeed ? <Dices size={16} /> : <Lock size={16} />}
                </button>
              </div>
            </Section>
            <Section label="CLIP skip">
              <div className="segmented">
                {[1, 2].map((n) => (
                  <button key={n} className={form.clip_skip === n ? "active" : ""} onClick={() => setForm({ clip_skip: n })}>
                    {n}
                  </button>
                ))}
              </div>
              <p className="hint">1 is the standard SDXL setting.</p>
            </Section>
          </div>
        )}
      </div>

      <div className="panel-footer">
        <button className="generate-btn" disabled={!canGenerate} onClick={onGenerate}>
          <Zap size={16} fill="currentColor" />
          Generate{form.count > 1 ? ` ×${form.count}` : ""}
        </button>
      </div>
    </aside>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="section">
      <div className="section-label">{label}</div>
      {children}
    </div>
  );
}

function Slider(props: { label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
  return (
    <div className="section">
      <div className="section-label slider-label">
        {props.label}
        <span className="slider-value">{props.value}</span>
      </div>
      <input
        type="range"
        className="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </div>
  );
}
