import { useCallback, useEffect, useMemo, useState } from "react";
import {
  api,
  isActive,
  type Asset,
  type Checkpoint,
  type Job,
  type MultiviewSupport,
  type Sampler,
  type Status,
  type TextureBaker,
} from "./api";
import { AssetsPanel, type Selection } from "./components/AssetsPanel";
import { Rail, TopBar, type Tool } from "./components/Chrome";
import { ImagePanel, SUPPORTED_ARCHS, sizeForArch, type ImageForm } from "./components/ImagePanel";
import { Model3DPanel, type Model3DForm, type Source3D } from "./components/Model3DPanel";
import { MODE_CFG, MultiviewPanel, type MultiviewForm, type Reference } from "./components/MultiviewPanel";
import { RemeshPanel, type RemeshForm } from "./components/RemeshPanel";
import { defaultFrontYaw, TexturePanel, type TextureForm, type TextureReference } from "./components/TexturePanel";
import { Viewer } from "./components/Viewer";
import { useJobs } from "./useJobs";

const PAGE = 60;

const DEFAULT_FORM: ImageForm = {
  prompt: "",
  negative: "lowres, worst quality, bad quality, bad anatomy, jpeg artifacts, watermark, signature",
  model: "",
  width: 1024,
  height: 1024,
  steps: 28,
  cfg: 6,
  sampler: "euler_a",
  seed: 0,
  randomSeed: true,
  clip_skip: 1,
  count: 1,
};

const DEFAULT_MV_FORM: MultiviewForm = {
  mode: "i2mv",
  prompt: "",
  imagePrompt: "",
  negative: "watermark, ugly, deformed, noisy, blurry, low contrast",
  model: "",
  steps: 40,
  cfg: MODE_CFG.i2mv,
  sampler: "euler",
  seed: 0,
  randomSeed: true,
  count: 1,
  removeBg: true,
  referenceScale: 1,
};

const DEFAULT_3D_FORM: Model3DForm = { removeBg: true, foregroundRatio: 0.85, resolution: 256 };

// MeshFlow's bundled config: 28 steps, image guidance 2.5.
const DEFAULT_REMESH_FORM: RemeshForm = {
  count: 1,
  fillHoles: false,
  steps: 28,
  seed: 0,
  randomSeed: true,
  useImage: true,
  removeBg: true,
  guidance: 2.5,
};

// Defaults from MV-Adapter's texture_i2tex script (CFG 3, 768 px views); 40 steps instead of 50 for speed.
const DEFAULT_TEXTURE_FORM: TextureForm = {
  mode: "ig2mv",
  prompt: "",
  model: "",
  removeBg: true,
  steps: 40,
  cfg: 3,
  sampler: "euler",
  seed: 0,
  randomSeed: true,
  referenceScale: 1,
  uvSize: 2048,
};

/** Form state remembered per browser; storage failures just mean defaults next time. */
function usePersistentForm<T extends object>(key: string, defaults: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const saved = localStorage.getItem(key);
      if (saved) return { ...defaults, ...JSON.parse(saved) };
    } catch {
      /* storage unavailable */
    }
    return defaults;
  });
  const update = useCallback(
    (patch: Partial<T>) =>
      setValue((v) => {
        const next = { ...v, ...patch };
        try {
          localStorage.setItem(key, JSON.stringify(next));
        } catch {
          /* storage unavailable */
        }
        return next;
      }),
    [key],
  );
  return [value, update, setValue] as const;
}

async function urlToDataUrl(url: string): Promise<string> {
  const blob = await (await fetch(url)).blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export default function App() {
  const [tool, setTool] = useState<Tool>("image");
  const [form, setForm, setFormState] = usePersistentForm<ImageForm>("trivexa.imageForm", DEFAULT_FORM);
  const [mvForm, setMvForm, setMvFormState] = usePersistentForm<MultiviewForm>("trivexa.multiviewForm", DEFAULT_MV_FORM);
  const [reference, setReference] = useState<Reference | null>(null);
  const [mvSupport, setMvSupport] = useState<MultiviewSupport | null>(null);
  const [form3d, setForm3d] = usePersistentForm<Model3DForm>("trivexa.model3dForm", DEFAULT_3D_FORM);
  const [source3d, setSource3d] = useState<Source3D | null>(null);
  const [tripoAvailable, setTripoAvailable] = useState(false);
  const [meshflowAvailable, setMeshflowAvailable] = useState(false);
  const [meshflowImageAvailable, setMeshflowImageAvailable] = useState(false);
  const [remeshForm, setRemeshForm] = usePersistentForm<RemeshForm>("trivexa.remeshForm", DEFAULT_REMESH_FORM);
  const [remeshModel, setRemeshModelState] = useState<Asset | null>(null);
  const [remeshReference, setRemeshReference] = useState<TextureReference | null>(null);
  const setRemeshModel = useCallback((a: Asset | null) => {
    setRemeshModelState(a);
    setRemeshReference(null);
  }, []);
  const [textureBaker, setTextureBaker] = useState<TextureBaker | null>(null);
  const [texForm, setTexForm, setTexFormState] = usePersistentForm<TextureForm>("trivexa.textureForm", DEFAULT_TEXTURE_FORM);
  const [texModel, setTexModelState] = useState<Asset | null>(null);
  const [texYaw, setTexYaw] = useState(0);
  const [texReference, setTexReference] = useState<TextureReference | null>(null);
  const setTexModel = useCallback((a: Asset | null) => {
    setTexModelState(a);
    setTexReference(null);
    if (a) setTexYaw(defaultFrontYaw(a));
  }, []);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [samplers, setSamplers] = useState<Sampler[]>([]);
  const [status, setStatus] = useState<Status | null>(null);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [selection, setSelection] = useState<Selection>(null);
  const [tab, setTab] = useState<"assets" | "details">("assets");
  const [submitting, setSubmitting] = useState(false);
  const [toasts, setToasts] = useState<{ id: number; msg: string; error?: boolean }[]>([]);

  const notify = useCallback((msg: string, error = false) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg, error }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), error ? 6000 : 2500);
  }, []);

  const onJobFinished = useCallback(
    (job: Job, asset: Asset | null) => {
      if (job.status === "error") notify(`Generation failed: ${job.error}`, true);
      if (!asset) return;
      if (!favoritesOnly) setAssets((list) => (list.some((a) => a.id === asset.id) ? list : [asset, ...list]));
      setSelection((sel) => (sel?.kind === "job" && sel.id === job.id ? { kind: "asset", id: asset.id } : sel));
    },
    [favoritesOnly, notify],
  );
  const { jobs, connected } = useJobs(onJobFinished);

  // -- data loading ---------------------------------------------------------

  // Re-checked when switching tools, so newly downloaded adapter weights show up without a reload.
  useEffect(() => {
    api.models().then(({ checkpoints, multiview, model3d, texture }) => {
      setTextureBaker(texture);
      setCheckpoints(checkpoints);
      setMvSupport(multiview);
      setTripoAvailable(model3d.triposr);
      setMeshflowAvailable(model3d.meshflow);
      setMeshflowImageAvailable(model3d.meshflow_image);
      const firstSdxl = checkpoints.find((c) => c.arch === "sdxl")?.id ?? "";
      setMvFormState((f) => (checkpoints.some((c) => c.id === f.model && c.arch === "sdxl") ? f : { ...f, model: firstSdxl }));
      setTexFormState((f) => (checkpoints.some((c) => c.id === f.model && c.arch === "sdxl") ? f : { ...f, model: firstSdxl }));
      setFormState((f) => {
        if (checkpoints.some((c) => c.id === f.model && (SUPPORTED_ARCHS as string[]).includes(c.arch))) return f;
        // Default to the first SDXL checkpoint, falling back to SD 1.5.
        for (const arch of SUPPORTED_ARCHS) {
          const first = checkpoints.find((c) => c.arch === arch);
          if (first) return { ...f, model: first.id, ...sizeForArch(f.width, f.height, arch) };
        }
        return { ...f, model: "" };
      });
    }, (e) => notify(`Could not load models: ${e.message}`, true));
  }, [notify, tool, setFormState, setMvFormState, setTexFormState]);

  useEffect(() => {
    api.samplers().then(({ samplers }) => setSamplers(samplers));
  }, []);

  useEffect(() => {
    const tick = () => api.status().then(setStatus, () => setStatus(null));
    tick();
    const t = setInterval(tick, 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    api.assets({ favorite: favoritesOnly, limit: PAGE }).then(({ assets }) => {
      setAssets(assets);
      setHasMore(assets.length === PAGE);
    });
  }, [favoritesOnly]);

  const loadMore = () =>
    api.assets({ favorite: favoritesOnly, limit: PAGE, offset: assets.length }).then(({ assets: more }) => {
      setAssets((list) => [...list, ...more.filter((m) => !list.some((a) => a.id === m.id))]);
      setHasMore(more.length === PAGE);
    });

  // -- derived --------------------------------------------------------------

  const activeJobs = useMemo(
    () => Object.values(jobs).filter(isActive).sort((a, b) => b.created_at - a.created_at),
    [jobs],
  );
  const queueOrder = useMemo(() => [...activeJobs].reverse().map((j) => j.id), [activeJobs]);

  const selectedJob = selection?.kind === "job" ? jobs[selection.id] ?? null : null;
  const selectedAsset =
    selection?.kind === "asset"
      ? assets.find((a) => a.id === selection.id) ?? null
      : selectedJob?.asset ?? (selection === null ? assets[0] ?? null : null);

  // -- actions --------------------------------------------------------------

  const generate = async () => {
    setSubmitting(true);
    try {
      const { randomSeed, ...rest } = form;
      const { jobs: created } = await api.generateImage({ ...rest, seed: randomSeed ? -1 : form.seed });
      if (created[0]) setSelection({ kind: "job", id: created[0].id });
      setTab("assets");
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const generateMultiview = async () => {
    setSubmitting(true);
    try {
      const { randomSeed, removeBg, referenceScale, imagePrompt, ...rest } = mvForm;
      const { jobs: created } = await api.generateMultiview({
        ...rest,
        prompt: mvForm.mode === "i2mv" ? imagePrompt : mvForm.prompt,
        seed: randomSeed ? -1 : mvForm.seed,
        remove_bg: removeBg,
        reference_scale: referenceScale,
        reference_asset: mvForm.mode === "i2mv" && reference?.type === "asset" ? reference.asset.id : null,
        reference_image: mvForm.mode === "i2mv" && reference?.type === "upload" ? reference.dataUrl : null,
      });
      if (created[0]) setSelection({ kind: "job", id: created[0].id });
      setTab("assets");
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const generate3d = async () => {
    if (!source3d) return;
    setSubmitting(true);
    try {
      const { jobs: created } = await api.generate3d({
        source_asset: source3d.type === "asset" ? source3d.asset.id : null,
        source_view: source3d.type === "asset" ? source3d.view : 0,
        source_image: source3d.type === "upload" ? source3d.dataUrl : null,
        remove_bg: form3d.removeBg,
        foreground_ratio: form3d.foregroundRatio,
        mc_resolution: form3d.resolution,
      });
      if (created[0]) setSelection({ kind: "job", id: created[0].id });
      setTab("assets");
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const generateTexture = async () => {
    if (!texModel) return;
    setSubmitting(true);
    try {
      const { jobs: created } = await api.generateTexture({
        source_asset: texModel.id,
        model: texForm.model,
        mode: texForm.mode,
        reference_asset: texReference?.type === "asset" ? texReference.asset.id : null,
        reference_image: texReference?.type === "upload" ? texReference.dataUrl : null,
        remove_bg: texForm.removeBg,
        prompt: texForm.mode === "tg2mv" ? texForm.prompt : texForm.prompt || "high quality",
        steps: texForm.steps,
        cfg: texForm.cfg,
        sampler: texForm.sampler,
        seed: texForm.randomSeed ? -1 : texForm.seed,
        reference_scale: texForm.referenceScale,
        front_yaw: texYaw,
        uv_size: texForm.uvSize,
      });
      if (created[0]) setSelection({ kind: "job", id: created[0].id });
      setTab("assets");
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const generateRemesh = async () => {
    if (!remeshModel) return;
    setSubmitting(true);
    try {
      const { jobs: created } = await api.generateRemesh({
        source_asset: remeshModel.id,
        steps: remeshForm.steps,
        seed: remeshForm.randomSeed ? -1 : remeshForm.seed,
        count: remeshForm.count,
        fill_holes: remeshForm.fillHoles,
        use_image: remeshForm.useImage && meshflowImageAvailable,
        reference_asset: remeshReference?.type === "asset" ? remeshReference.asset.id : null,
        reference_image: remeshReference?.type === "upload" ? remeshReference.dataUrl : null,
        remove_bg: remeshForm.removeBg,
        guidance: remeshForm.guidance,
      });
      if (created[0]) setSelection({ kind: "job", id: created[0].id });
      setTab("assets");
    } catch (e) {
      notify((e as Error).message, true);
    } finally {
      setSubmitting(false);
    }
  };

  const makeRemesh = (a: Asset) => {
    setRemeshModel(a);
    setTool("remesh");
  };

  const makeTexture = (a: Asset) => {
    setTexModel(a);
    setTool("texture");
  };

  const make3D = (a: Asset) => {
    setSource3d({ type: "asset", asset: a, view: 0 });
    setTool("model3d");
  };

  // The 3D viewer renders a proper thumbnail the first time a model is opened.
  const saveModelSnapshot = async (a: Asset, dataUrl: string) => {
    try {
      const updated = await api.setThumbnail(a.id, dataUrl);
      setAssets((list) => list.map((x) => (x.id === a.id ? updated : x)));
    } catch {
      /* keep the input-image thumbnail */
    }
  };

  const makeMultiview = (a: Asset) => {
    setReference({ type: "asset", asset: a });
    setMvForm({ mode: "i2mv", cfg: mvForm.mode === "i2mv" ? mvForm.cfg : MODE_CFG.i2mv });
    setTool("multiview");
  };

  const toggleFavorite = async (a: Asset) => {
    try {
      const updated = await api.setFavorite(a.id, !a.favorite);
      setAssets((list) =>
        favoritesOnly && !updated.favorite ? list.filter((x) => x.id !== a.id) : list.map((x) => (x.id === a.id ? updated : x)),
      );
    } catch (e) {
      notify((e as Error).message, true);
    }
  };

  const deleteAsset = async (a: Asset) => {
    const what = { image: "this image", multiview: "these views", model3d: "this 3D model" }[a.kind];
    if (!confirm(`Delete ${what} from disk?`)) return;
    try {
      await api.deleteAsset(a.id);
      const idx = assets.findIndex((x) => x.id === a.id);
      const rest = assets.filter((x) => x.id !== a.id);
      setAssets(rest);
      const next = rest[Math.min(idx, rest.length - 1)];
      setSelection(next ? { kind: "asset", id: next.id } : null);
    } catch (e) {
      notify((e as Error).message, true);
    }
  };

  const reuse = async (a: Asset) => {
    const p = a.params;
    if (a.kind === "multiview" && p.mode) {
      setMvForm({
        mode: p.mode,
        ...(p.mode === "i2mv" ? { imagePrompt: p.prompt } : { prompt: p.prompt }),
        negative: p.negative,
        model: checkpoints.some((c) => c.id === p.model) ? p.model : mvForm.model,
        steps: p.steps,
        cfg: p.cfg,
        sampler: p.sampler,
        seed: p.seed,
        randomSeed: false,
        removeBg: p.remove_bg ?? true,
        referenceScale: p.reference_scale ?? 1,
      });
      setTool("multiview");
      if (p.mode === "i2mv") {
        try {
          if (p.reference_asset) setReference({ type: "asset", asset: await api.asset(p.reference_asset) });
          else if (a.reference_url) setReference({ type: "upload", dataUrl: await urlToDataUrl(a.reference_url), name: "reference.png" });
        } catch {
          notify("The original reference image is no longer available", true);
        }
      }
      notify("Settings loaded — seed locked");
      return;
    }
    setTool("image");
    setForm({
      prompt: p.prompt,
      negative: p.negative,
      model: checkpoints.some((c) => c.id === p.model) ? p.model : form.model,
      width: p.width,
      height: p.height,
      steps: p.steps,
      cfg: p.cfg,
      sampler: p.sampler,
      seed: p.seed,
      randomSeed: false,
      clip_skip: p.clip_skip,
    });
    notify("Settings loaded — seed locked");
  };

  const cancel = (id: string) => api.cancelJob(id).catch((e) => notify(e.message, true));

  // Arrow keys step through the gallery when not typing.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      const idx = selectedAsset ? assets.findIndex((a) => a.id === selectedAsset.id) : -1;
      const next = assets[Math.max(0, Math.min(assets.length - 1, idx + (e.key === "ArrowRight" ? 1 : -1)))];
      if (next) setSelection({ kind: "asset", id: next.id });
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [assets, selectedAsset]);

  return (
    <div className="app">
      <TopBar
        tool={tool}
        status={status}
        connected={connected}
        onUnload={() => api.unload().then(() => api.status().then(setStatus))}
      />
      <div className="workspace">
        <Rail active={tool} onSelect={setTool} />
        {tool === "remesh" ? (
          <RemeshPanel
            form={remeshForm}
            setForm={setRemeshForm}
            model={remeshModel}
            setModel={setRemeshModel}
            reference={remeshReference}
            setReference={setRemeshReference}
            selected={selectedAsset}
            available={meshflowAvailable}
            imageAvailable={meshflowImageAvailable}
            bgAvailable={mvSupport?.bg_removal ?? false}
            onGenerate={generateRemesh}
            submitting={submitting}
            notify={notify}
          />
        ) : tool === "texture" ? (
          <TexturePanel
            form={texForm}
            setForm={setTexForm}
            model={texModel}
            setModel={setTexModel}
            frontYaw={texYaw}
            setFrontYaw={setTexYaw}
            reference={texReference}
            setReference={setTexReference}
            selected={selectedAsset}
            checkpoints={checkpoints}
            samplers={samplers}
            support={mvSupport}
            baker={textureBaker}
            onGenerate={generateTexture}
            submitting={submitting}
            notify={notify}
          />
        ) : tool === "model3d" ? (
          <Model3DPanel
            form={form3d}
            setForm={setForm3d}
            source={source3d}
            setSource={setSource3d}
            selected={selectedAsset}
            available={tripoAvailable}
            bgAvailable={mvSupport?.bg_removal ?? false}
            onGenerate={generate3d}
            submitting={submitting}
            notify={notify}
          />
        ) : tool === "image" ? (
          <ImagePanel
            form={form}
            setForm={setForm}
            checkpoints={checkpoints}
            samplers={samplers}
            onGenerate={generate}
            submitting={submitting}
          />
        ) : (
          <MultiviewPanel
            form={mvForm}
            setForm={setMvForm}
            reference={reference}
            setReference={setReference}
            selectedImage={selectedAsset?.kind === "image" ? selectedAsset : null}
            checkpoints={checkpoints}
            samplers={samplers}
            support={mvSupport}
            onGenerate={generateMultiview}
            submitting={submitting}
            notify={notify}
          />
        )}
        <Viewer
          asset={selectedJob && selectedJob.status !== "done" ? null : selectedAsset}
          job={selectedJob}
          queuePosition={selectedJob ? queueOrder.indexOf(selectedJob.id) : -1}
          onCancel={cancel}
          onToggleFavorite={toggleFavorite}
          onDelete={deleteAsset}
          onReuse={reuse}
          onMakeMultiview={makeMultiview}
          onMake3D={make3D}
          onTexture={makeTexture}
          onRemesh={makeRemesh}
          onModelSnapshot={saveModelSnapshot}
          notify={notify}
        />
        <AssetsPanel
          tab={tab}
          setTab={setTab}
          assets={assets}
          activeJobs={activeJobs}
          selection={selection ?? (selectedAsset ? { kind: "asset", id: selectedAsset.id } : null)}
          onSelect={setSelection}
          favoritesOnly={favoritesOnly}
          setFavoritesOnly={setFavoritesOnly}
          onToggleFavorite={toggleFavorite}
          hasMore={hasMore}
          onLoadMore={loadMore}
          detailAsset={selectedAsset}
        />
      </div>
      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.error ? "error" : ""}`}>
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  );
}
