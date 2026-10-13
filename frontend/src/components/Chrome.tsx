import { Box, Cpu, Image as ImageIcon, Layers, PaintBucket, Power } from "lucide-react";
import type { Status } from "../api";
import { Logo } from "./Logo";

const GB = 1024 ** 3;

export type Tool = "image" | "multiview" | "model3d" | "texture";

export function TopBar(props: { tool: Tool; status: Status | null; connected: boolean; onUnload: () => void }) {
  const { status, connected, onUnload } = props;
  const gpu = status?.gpu;
  return (
    <header className="topbar">
      <div className="brand">
        <Logo />
        <span>TRIVEXA</span>
      </div>
      <div className="workspace-pill">{TOOLS.find((t) => t.id === props.tool)?.label} Workspace</div>
      <div className="topbar-right">
        {!connected && <span className="pill warn">Reconnecting…</span>}
        {status && status.pending_jobs > 0 && <span className="pill">{status.pending_jobs} in queue</span>}
        {status?.loaded_model && (
          <span className="pill" title="Model loaded in VRAM">
            {status.loaded_model}
            <button className="pill-btn" onClick={onUnload} title="Unload model and free VRAM">
              <Power size={12} />
            </button>
          </span>
        )}
        {gpu && (
          <span className="pill" title={gpu.name}>
            <Cpu size={13} />
            {((gpu.vram_total - gpu.vram_free) / GB).toFixed(1)} / {(gpu.vram_total / GB).toFixed(0)} GB
          </span>
        )}
      </div>
    </header>
  );
}

const TOOLS = [
  { id: "image", label: "Image", icon: ImageIcon, enabled: true },
  { id: "multiview", label: "Multiview", icon: Layers, enabled: true },
  { id: "model3d", label: "3D Model", icon: Box, enabled: true },
  { id: "texture", label: "Texture", icon: PaintBucket, enabled: true },
];

export function Rail({ active, onSelect }: { active: Tool; onSelect: (t: Tool) => void }) {
  return (
    <nav className="rail">
      {TOOLS.map(({ id, label, icon: Icon, enabled }) => (
        <button
          key={id}
          className={`rail-item ${active === id ? "active" : ""}`}
          disabled={!enabled}
          title={enabled ? label : `${label} — coming soon`}
          onClick={() => onSelect(id as Tool)}
        >
          <Icon size={18} />
          <span>{label}</span>
          {!enabled && <em>soon</em>}
        </button>
      ))}
    </nav>
  );
}
