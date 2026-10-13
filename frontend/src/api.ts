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
  model_name?: string;
  sampler_label?: string;
}

export interface Asset {
  id: string;
  kind: "image" | "model3d";
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
  params: ImageParams;
  favorite: boolean;
}

export type JobStatus = "queued" | "loading" | "running" | "done" | "error" | "cancelled";

export interface Job {
  id: string;
  kind: string;
  params: ImageParams;
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
  models: () => request<{ checkpoints: Checkpoint[] }>("/api/models"),
  samplers: () => request<{ samplers: Sampler[] }>("/api/samplers"),
  generateImage: (params: Omit<ImageParams, "model_name" | "sampler_label"> & { count: number }) =>
    request<{ jobs: Job[] }>("/api/generate/image", { method: "POST", body: JSON.stringify(params) }),
  cancelJob: (id: string) => request(`/api/jobs/${id}/cancel`, { method: "POST" }),
  assets: (opts: { favorite?: boolean; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    if (opts.favorite) q.set("favorite", "true");
    q.set("limit", String(opts.limit ?? 60));
    q.set("offset", String(opts.offset ?? 0));
    return request<{ assets: Asset[] }>(`/api/assets?${q}`);
  },
  setFavorite: (id: string, favorite: boolean) =>
    request<Asset>(`/api/assets/${id}`, { method: "PATCH", body: JSON.stringify({ favorite }) }),
  deleteAsset: (id: string) => request(`/api/assets/${id}`, { method: "DELETE" }),
};
