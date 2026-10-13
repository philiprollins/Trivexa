export interface Checkpoint {
  id: string;
  name: string;
  arch: "sdxl" | "sd15" | "unknown";
  size: number;
}

export interface Sampler {
  id: string;
  label: string;
}

export interface ImageParams {
  prompt: string;
  negative: string;
  model: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  seed: number;
  clip_skip: number;
  arch?: "sdxl" | "sd15";
  model_name?: string;
  sampler_label?: string;
}

export interface MultiviewParams {
  mode: "t2mv" | "i2mv";
  prompt: string;
  negative: string;
  model: string;
  steps: number;
  cfg: number;
  sampler: string;
  seed: number;
  reference_asset?: string | null;
  remove_bg?: boolean;
  reference_scale?: number;
  azimuths?: number[];
  model_name?: string;
  sampler_label?: string;
}

export interface Model3DParams {
  source_asset?: string | null;
  source_view?: number;
  remove_bg: boolean;
  foreground_ratio: number;
  mc_resolution: number;
  engine: string;
  vertices?: number;
  faces?: number;
}

export interface Asset {
  id: string;
  kind: "image" | "multiview" | "model3d";
  created_at: number;
  file: string;
  url: string;
  thumb_url: string;
  width: number;
  height: number;
  prompt: string;
  negative: string;
  model: string;
  seed: number;
  params: ImageParams & Partial<MultiviewParams> & Partial<Model3DParams>;
  favorite: boolean;
  parent_id: string | null;
  views?: { azimuth: number; url: string }[];
  reference_url?: string;
  model_urls?: { glb: string; obj: string };
  input_url?: string;
  thumb_rendered?: boolean;
}

export type JobStatus = "queued" | "loading" | "running" | "done" | "error" | "cancelled";

export interface Job {
  id: string;
  kind: "image" | "multiview" | "model3d";
  params: ImageParams & Partial<MultiviewParams> & Partial<Model3DParams>;
  status: JobStatus;
  step: number;
  total: number;
  preview: string | null;
  asset: Asset | null;
  error: string | null;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
}

export interface Status {
  gpu: { name: string; vram_free: number; vram_total: number } | null;
  loaded_model: string | null;
  pending_jobs: number;
  models_dir: string;
  outputs_dir: string;
}

export interface MultiviewSupport {
  t2mv: boolean;
  i2mv: boolean;
  bg_removal: boolean;
}

export type MultiviewRequest = Omit<MultiviewParams, "azimuths" | "model_name" | "sampler_label"> & {
  count: number;
  reference_image?: string | null;
};

export const isActive = (j: Job) => j.status === "queued" || j.status === "loading" || j.status === "running";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: init?.body ? { "Content-Type": "application/json" } : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export const api = {
  status: () => request<Status>("/api/status"),
  unload: () => request("/api/unload", { method: "POST" }),
  models: () =>
    request<{ checkpoints: Checkpoint[]; multiview: MultiviewSupport; model3d: { triposr: boolean } }>("/api/models"),
  samplers: () => request<{ samplers: Sampler[] }>("/api/samplers"),
  generateImage: (params: Omit<ImageParams, "model_name" | "sampler_label" | "arch"> & { count: number }) =>
    request<{ jobs: Job[] }>("/api/generate/image", { method: "POST", body: JSON.stringify(params) }),
  generateMultiview: (params: MultiviewRequest) =>
    request<{ jobs: Job[] }>("/api/generate/multiview", { method: "POST", body: JSON.stringify(params) }),
  generate3d: (params: {
    source_asset?: string | null;
    source_view?: number;
    source_image?: string | null;
    remove_bg: boolean;
    foreground_ratio: number;
    mc_resolution: number;
  }) => request<{ jobs: Job[] }>("/api/generate/3d", { method: "POST", body: JSON.stringify(params) }),
  setThumbnail: (id: string, image: string) =>
    request<Asset>(`/api/assets/${id}/thumbnail`, { method: "POST", body: JSON.stringify({ image }) }),
  cancelJob: (id: string) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  assets: (opts: { favorite?: boolean; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.favorite) q.set("favorite", "true");
    q.set("limit", String(opts.limit ?? 60));
    q.set("offset", String(opts.offset ?? 0));
    return request<{ assets: Asset[] }>(`/api/assets?${q}`);
  },
  asset: (id: string) => request<Asset>(`/api/assets/${id}`),
  setFavorite: (id: string, favorite: boolean) =>
    request<Asset>(`/api/assets/${id}`, { method: "PATCH", body: JSON.stringify({ favorite }) }),
  deleteAsset: (id: string) => request(`/api/assets/${id}`, { method: "DELETE" }),
};
