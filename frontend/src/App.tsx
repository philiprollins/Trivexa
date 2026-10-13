import { useCallback, useEffect, useMemo, useState } from "react";
import { api, isActive, type Asset, type Checkpoint, type Job, type Sampler, type Status } from "./api";
import { AssetsPanel, type Selection } from "./components/AssetsPanel";
import { Rail, TopBar } from "./components/Chrome";
import { ImagePanel, type ImageForm } from "./components/ImagePanel";
import { Viewer } from "./components/Viewer";
import { useJobs } from "./useJobs";

const FORM_KEY = "trivexa.imageForm";
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

function loadForm(): ImageForm {
  try {
    const saved = localStorage.getItem(FORM_KEY);
    if (saved) return { ...DEFAULT_FORM, ...JSON.parse(saved) };
  } catch {
    /* storage unavailable */
  }
  return DEFAULT_FORM;
}

export default function App() {
  const [form, setFormState] = useState<ImageForm>(loadForm);
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

  const setForm = useCallback((patch: Partial<ImageForm>) => {
    setFormState((f) => {
      const next = { ...f, ...patch };
      try {
        localStorage.setItem(FORM_KEY, JSON.stringify(next));
      } catch {
        /* storage unavailable */
      }
      return next;
    });
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

  useEffect(() => {
    api.models().then(({ checkpoints }) => {
      setCheckpoints(checkpoints);
      const sdxl = checkpoints.filter((c) => c.arch === "sdxl");
      setFormState((f) => (sdxl.some((c) => c.id === f.model) ? f : { ...f, model: sdxl[0]?.id ?? "" }));
    }, (e) => notify(`Could not load models: ${e.message}`, true));
    api.samplers().then(({ samplers }) => setSamplers(samplers));
  }, [notify]);

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
    if (!confirm("Delete this image from disk?")) return;
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

  const reuse = (a: Asset) => {
    const p = a.params;
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
        status={status}
        connected={connected}
        onUnload={() => api.unload().then(() => api.status().then(setStatus))}
      />
      <div className="workspace">
        <Rail active="image" />
        <ImagePanel
          form={form}
          setForm={setForm}
          checkpoints={checkpoints}
          samplers={samplers}
          onGenerate={generate}
          submitting={submitting}
        />
        <Viewer
          asset={selectedJob && selectedJob.status !== "done" ? null : selectedAsset}
          job={selectedJob}
          queuePosition={selectedJob ? queueOrder.indexOf(selectedJob.id) : -1}
          onCancel={cancel}
          onToggleFavorite={toggleFavorite}
          onDelete={deleteAsset}
          onReuse={reuse}
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
