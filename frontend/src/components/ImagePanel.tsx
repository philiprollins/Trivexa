import { ChevronDown, Sparkles, Zap } from "lucide-react";
import { useState } from "react";
import type { Checkpoint, ImageParams, Sampler } from "../api";
import { SeedInput, Section, Slider } from "./controls";

export type ImageForm = Omit<ImageParams, "model_name" | "sampler_label" | "arch"> & { count: number; randomSeed: boolean };

type Arch = "sdxl" | "sd15";
export const SUPPORTED_ARCHS: Arch[] = ["sdxl", "sd15"];
const ARCH_LABELS: Record<Arch, string> = { sdxl: "SDXL", sd15: "SD 1.5" };

// Sizes near each architecture's native training resolution (1024² for SDXL, 512² for SD 1.5).
export const ASPECTS: Record<Arch, { label: string; w: number; h: number }[]> = {
  sdxl: [
    { label: "1:1", w: 1024, h: 1024 },
    { label: "3:4", w: 896, h: 1152 },
    { label: "2:3", w: 832, h: 1216 },
    { label: "9:16", w: 768, h: 1344 },
    { label: "4:3", w: 1152, h: 896 },
    { label: "3:2", w: 1216, h: 832 },
    { label: "16:9", w: 1344, h: 768 },
  ],
  sd15: [
    { label: "1:1", w: 512, h: 512 },
    { label: "3:4", w: 576, h: 768 },
    { label: "2:3", w: 512, h: 768 },
    { label: "9:16", w: 448, h: 800 },
    { label: "4:3", w: 768, h: 576 },
    { label: "3:2", w: 768, h: 512 },
    { label: "16:9", w: 800, h: 448 },
  ],
};

export const archOf = (checkpoints: Checkpoint[], id: string) =>
  checkpoints.find((c) => c.id === id)?.arch as Arch | undefined;

/** Carry the chosen aspect ratio over to another architecture's native size. */
export function sizeForArch(width: number, height: number, to: Arch): { width: number; height: number } {
  const label = Object.values(ASPECTS)
    .flat()
    .find((a) => a.w === width && a.h === height)?.label;
  const preset = ASPECTS[to].find((a) => a.label === label) ?? ASPECTS[to][0];
  return { width: preset.w, height: preset.h };
}

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
  const arch = archOf(checkpoints, form.model) ?? "sdxl";
  const supported = checkpoints.filter((c) => (SUPPORTED_ARCHS as string[]).includes(c.arch));
  const canGenerate = form.prompt.trim().length > 0 && supported.some((c) => c.id === form.model) && !submitting;

  const selectModel = (id: string) => {
    const next = archOf(checkpoints, id);
    setForm(next && next !== arch ? { model: id, ...sizeForArch(form.width, form.height, next) } : { model: id });
  };

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
          <select className="select" value={form.model} onChange={(e) => selectModel(e.target.value)}>
            {supported.length === 0 && <option value="">No supported checkpoints found</option>}
            {SUPPORTED_ARCHS.map((a) => {
              const group = supported.filter((c) => c.arch === a);
              return (
                group.length > 0 && (
                  <optgroup key={a} label={ARCH_LABELS[a]}>
                    {group.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name}
                      </option>
                    ))}
                  </optgroup>
                )
              );
            })}
          </select>
          <p className="hint">{ARCH_LABELS[arch]} · native {arch === "sdxl" ? "1024" : "512"} px</p>
        </Section>

        <Section label="Aspect ratio">
          <div className="aspect-grid">
            {ASPECTS[arch].map((a) => {
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
            <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
            <Section label="CLIP skip">
              <div className="segmented">
                {[1, 2].map((n) => (
                  <button key={n} className={form.clip_skip === n ? "active" : ""} onClick={() => setForm({ clip_skip: n })}>
                    {n}
                  </button>
                ))}
              </div>
              <p className="hint">1 is standard. Many anime-style SD 1.5 models are tuned for 2.</p>
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
