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
} from "./api";
import { AssetsPanel, type Selection } from "./components/AssetsPanel";
import { Rail, TopBar, type Tool } from "./components/Chrome";
import { ImagePanel, SUPPORTED_ARCHS, sizeForArch, type ImageForm } from "./components/ImagePanel";
import { MODE_CFG, MultiviewPanel, type MultiviewForm, type Reference } from "./components/MultiviewPanel";
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
    api.models().then(({ checkpoints, multiview }) => {
      setCheckpoints(checkpoints);
      setMvSupport(multiview);
      const firstSdxl = checkpoints.find((c) => c.arch === "sdxl")?.id ?? "";
      setMvFormState((f) => (checkpoints.some((c) => c.id === f.model && c.arch === "sdxl") ? f : { ...f, model: firstSdxl }));
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
  }, [notify, tool, setFormState, setMvFormState]);

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
    if (!confirm(a.kind === "multiview" ? "Delete these views from disk?" : "Delete this image from disk?")) return;
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
        {tool === "image" ? (
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
