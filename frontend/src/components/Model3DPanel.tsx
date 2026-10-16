import { Box, ChevronDown, ImagePlus, X, Zap } from "lucide-react";
import { useRef, useState } from "react";
import type { Asset } from "../api";
import { SeedInput, Section, Segmented, Slider } from "./controls";

export type Engine3D = "triposr" | "triposg" | "pixal3d";

export interface Model3DForm {
  engine: Engine3D;
  removeBg: boolean;
  // TripoSR
  foregroundRatio: number;
  resolution: number;
  // TripoSG
  octreeDepth: 8 | 9;
  faceLimit: number; // 0 = keep every face
  steps: number;
  guidance: number;
  seed: number; // TripoSG and Pixal3D
  randomSeed: boolean;
  // Pixal3D
  p3dResolution: 1024 | 1536;
  p3dFov: number; // degrees; 0 = estimate with MoGe-2
  p3dTextureSize: 1024 | 2048 | 4096;
  p3dFaceLimit: number;
  p3dSteps: number;
  p3dGuidance: number;
  p3dMultiview: boolean; // with a multiview set as the source: condition on several of its views
  p3dViews: number[]; // azimuths picked for that (the front is always used)
}

const FACE_LIMITS = [
  { value: 50_000, label: "50k" },
  { value: 200_000, label: "200k" },
  { value: 500_000, label: "500k" },
  { value: 0, label: "Full" },
];

const P3D_FACE_LIMITS = [
  { value: 100_000, label: "100k" },
  { value: 200_000, label: "200k" },
  { value: 500_000, label: "500k" },
  { value: 1_000_000, label: "1M" },
];

export type Source3D =
  | { type: "asset"; asset: Asset; view: number }
  | { type: "upload"; dataUrl: string; name: string };

interface Props {
  form: Model3DForm;
  setForm: (f: Partial<Model3DForm>) => void;
  source: Source3D | null;
  setSource: (s: Source3D | null) => void;
  selected: Asset | null; // image or multiview asset currently selected in the gallery
  available: Record<Engine3D, boolean>;
  pixal3dNote: string | null; // why Pixal3D is unavailable
  pixal3dMv: boolean; // its multi-view weights are present
  bgAvailable: boolean;
  onGenerate: () => void;
  submitting: boolean;
  notify: (msg: string, error?: boolean) => void;
}

const MAX_UPLOAD = 20 * 1024 * 1024;

/** The views a Pixal3D multi-view request uses, or null for a single-image request. */
export function multiviewAzimuths(form: Model3DForm, source: Source3D | null, mvAvailable: boolean): number[] | null {
  if (form.engine !== "pixal3d" || !form.p3dMultiview || !mvAvailable) return null;
  if (source?.type !== "asset" || source.asset.kind !== "multiview" || !source.asset.views) return null;
  const have = source.asset.views.map((v) => v.azimuth);
  const picked = have.filter((a) => a === 0 || form.p3dViews.includes(a));
  return picked.includes(0) && picked.length >= 2 ? picked : null;
}

const viewLabel = (azimuth: number) => (azimuth === 0 ? "Front" : `${azimuth}°`);

export function sourcePreview(s: Source3D): string {
  if (s.type === "upload") return s.dataUrl;
  if (s.asset.kind === "multiview") return s.asset.views?.find((v) => v.azimuth === s.view)?.url ?? s.asset.thumb_url;
  return s.asset.thumb_url;
}

export function Model3DPanel(p: Props) {
  const { form, setForm, source } = p;
  const [dragging, setDragging] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const sg = form.engine === "triposg";
  const p3d = form.engine === "pixal3d";
  const mvSource = source?.type === "asset" && source.asset.kind === "multiview" && source.asset.views ? source.asset : null;
  const offerViews = p3d && p.pixal3dMv && mvSource !== null;
  const severalViews = offerViews && form.p3dMultiview;
  const mvViews = multiviewAzimuths(form, source, p.pixal3dMv);
  const canGenerate =
    p.available[form.engine] && source !== null && !(form.removeBg && !p.bgAvailable) && !p.submitting &&
    !(severalViews && !mvViews);

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

        {offerViews && (
          <Section label="Views">
            <Segmented
              options={[
                { value: "several", label: "Several views" },
                { value: "one", label: "One view" },
              ]}
              value={form.p3dMultiview ? "several" : "one"}
              onChange={(v) => setForm({ p3dMultiview: v === "several" })}
            />
            {severalViews && (
              <>
                <div className="segmented view-picker">
                  {mvSource!.views!.map(({ azimuth }) => {
                    const on = azimuth === 0 || form.p3dViews.includes(azimuth);
                    return (
                      <button
                        key={azimuth}
                        className={on ? "active" : ""}
                        disabled={azimuth === 0}
                        title={azimuth === 0 ? "The front view is always used: the others are placed relative to it" : undefined}
                        onClick={() =>
                          setForm({ p3dViews: on ? form.p3dViews.filter((a) => a !== azimuth) : [...form.p3dViews, azimuth] })
                        }
                      >
                        {viewLabel(azimuth)}
                      </button>
                    );
                  })}
                </div>
                <p className="hint">
                  Pixal3D's multi-view models project every picked view onto the shape through its own camera.
                  Front, sides and back (0/90/180/270°) is upstream's own rig.
                  {!mvViews && " Pick at least one view besides the front."}
                </p>
              </>
            )}
          </Section>
        )}

        {mvSource && !severalViews && source?.type === "asset" && (
          <Section label="View to reconstruct from">
            <select
              className="select"
              value={source.view}
              onChange={(e) => p.setSource({ ...source, view: Number(e.target.value) })}
            >
              {mvSource.views!.map((v) => (
                <option key={v.azimuth} value={v.azimuth}>
                  {v.azimuth === 0 ? "Front 0°" : `${v.azimuth}°`}
                </option>
              ))}
            </select>
            <p className="hint">
              {p3d && p.pixal3dMv
                ? "Reconstructs from this one image; the front view usually works best."
                : "Every engine reconstructs from a single image; the front view usually works best."}
            </p>
          </Section>
        )}

        <Section label="Engine">
          <Segmented
            options={[
              { value: "pixal3d", label: "Pixal3D" },
              { value: "triposg", label: "TripoSG" },
              { value: "triposr", label: "TripoSR" },
            ]}
            value={form.engine}
            onChange={(engine) => setForm({ engine })}
          />
          <div className="engine-card">
            {p3d ? (
              <span>
                Pixel-aligned shape and PBR texture in one pass (TRELLIS.2 backbone) · closest to the input image ·
                textured GLB · ~1–2 min
              </span>
            ) : sg ? (
              <span>
                High-fidelity shape from a 1.5B flow model · sharp detail and clean surfaces · no colour (texture it
                next) · ~30–60 s
              </span>
            ) : (
              <span>Fast single-image reconstruction · vertex colours · soft shapes · good for quick prototypes</span>
            )}
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

        {p3d ? (
          <>
            <Section label="Resolution">
              <Segmented
                options={[
                  { value: 1024, label: "1024" },
                  { value: 1536, label: "1536" },
                ]}
                value={form.p3dResolution}
                onChange={(p3dResolution) => setForm({ p3dResolution })}
              />
              <p className="hint">Voxel grid of the final shape. 1536 is upstream's default; 1024 is quicker.</p>
            </Section>

            <Section label="Face limit">
              <Segmented options={P3D_FACE_LIMITS} value={form.p3dFaceLimit} onChange={(p3dFaceLimit) => setForm({ p3dFaceLimit })} />
            </Section>

            <Section label="Texture size">
              <Segmented
                options={[1024, 2048, 4096].map((n) => ({ value: n as 1024 | 2048 | 4096, label: n }))}
                value={form.p3dTextureSize}
                onChange={(p3dTextureSize) => setForm({ p3dTextureSize })}
              />
            </Section>

            <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
              Camera &amp; sampling
              <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
            </button>
            {showAdvanced && (
              <div className="advanced">
                <Section label="Camera field of view">
                  <Segmented
                    options={[
                      { value: "auto", label: severalViews ? "Rig (20°)" : "Estimate" },
                      { value: "manual", label: "Set" },
                    ]}
                    value={form.p3dFov ? "manual" : "auto"}
                    onChange={(v) => setForm({ p3dFov: v === "manual" ? 30 : 0 })}
                  />
                  {form.p3dFov > 0 && (
                    <Slider label="FOV (°)" min={5} max={90} step={1} value={form.p3dFov} onChange={(p3dFov) => setForm({ p3dFov })} />
                  )}
                  <p className="hint">
                    {severalViews
                      ? "Pixal3D projects each view through its camera. Multiview sets are orthographic, which a narrow FOV approximates; 20° is upstream's rig."
                      : "Pixal3D projects the image through the camera that took it. MoGe-2 estimates its FOV; set a narrow one (~12°) if the shape looks stretched in depth."}
                  </p>
                </Section>
                <Slider label="Steps" min={4} max={50} step={1} value={form.p3dSteps} onChange={(p3dSteps) => setForm({ p3dSteps })} />
                <Slider
                  label="Guidance"
                  min={1}
                  max={12}
                  step={0.5}
                  value={form.p3dGuidance}
                  onChange={(p3dGuidance) => setForm({ p3dGuidance })}
                />
                <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
              </div>
            )}
          </>
        ) : sg ? (
          <>
            <Section label="Mesh resolution">
              <Segmented
                options={[
                  { value: 8, label: "256" },
                  { value: 9, label: "512" },
                ]}
                value={form.octreeDepth}
                onChange={(octreeDepth) => setForm({ octreeDepth })}
              />
              <p className="hint">Surface grid size. 512 keeps fine detail; 256 is quicker and softer.</p>
            </Section>

            <Section label="Face limit">
              <Segmented options={FACE_LIMITS} value={form.faceLimit} onChange={(faceLimit) => setForm({ faceLimit })} />
              <p className="hint">Decimates the raw mesh (often over a million faces) to keep files and later steps light.</p>
            </Section>

            <button className="collapse-head" onClick={() => setShowAdvanced((v) => !v)}>
              Sampling settings
              <ChevronDown size={16} className={showAdvanced ? "rot" : ""} />
            </button>
            {showAdvanced && (
              <div className="advanced">
                <Slider label="Steps" min={10} max={100} step={1} value={form.steps} onChange={(steps) => setForm({ steps })} />
                <Slider
                  label="Guidance"
                  min={1}
                  max={12}
                  step={0.5}
                  value={form.guidance}
                  onChange={(guidance) => setForm({ guidance })}
                />
                <SeedInput seed={form.seed} random={form.randomSeed} onChange={(seed, randomSeed) => setForm({ seed, randomSeed })} />
              </div>
            )}
          </>
        ) : (
          <>
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
          </>
        )}

        {!p.available[form.engine] &&
          (p3d ? (
            <p className="notice">
              Pixal3D unavailable: {p.pixal3dNote ?? "unknown reason"}. It reads <code>TencentARC/Pixal3D</code>,{" "}
              <code>clip_vision/dino_v3_L_naf_fp32.safetensors</code> and{" "}
              <code>geometry_estimation/moge_2_vitl_normal_fp16.safetensors</code> from the models folder.
            </p>
          ) : sg ? (
            <p className="notice">
              TripoSG weights not found in <code>data/models/triposg</code>. Run{" "}
              <code>uv run python -m trivexa.download triposg</code>.
            </p>
          ) : (
            <p className="notice">
              TripoSR weights not found. Expected <code>config.yaml</code>, <code>model.ckpt</code> and{" "}
              <code>dino/config.json</code> in <code>models/triposr</code>.
            </p>
          ))}
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
