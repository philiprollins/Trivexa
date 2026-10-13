import { Dices, Lock } from "lucide-react";

export function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="section">
      <div className="section-label">{label}</div>
      {children}
    </div>
  );
}

export function Slider(props: { label: string; min: number; max: number; step: number; value: number; onChange: (v: number) => void }) {
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

export function SeedInput(props: { seed: number; random: boolean; onChange: (seed: number, random: boolean) => void }) {
  return (
    <Section label="Seed">
      <div className="seed-row">
        <input
          className="input"
          type="number"
          min={0}
          max={4294967295}
          disabled={props.random}
          value={props.random ? "" : props.seed}
          placeholder="Random"
          onChange={(e) => props.onChange(Math.max(0, Math.min(4294967295, Number(e.target.value) || 0)), false)}
        />
        <button
          className={`icon-btn ${props.random ? "active" : ""}`}
          title={props.random ? "Random seed each time (click to lock)" : "Seed locked (click for random)"}
          onClick={() => props.onChange(props.seed, !props.random)}
        >
          {props.random ? <Dices size={16} /> : <Lock size={16} />}
        </button>
      </div>
    </Section>
  );
}

export function Segmented<T extends string | number>(props: {
  options: { value: T; label: React.ReactNode; disabled?: boolean }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented">
      {props.options.map((o) => (
        <button
          key={String(o.value)}
          className={props.value === o.value ? "active" : ""}
          disabled={o.disabled}
          onClick={() => props.onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
