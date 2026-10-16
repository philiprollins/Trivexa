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
  source_views?: number[]; // Pixal3D multi-view input
  remove_bg: boolean;
  foreground_ratio?: number; // TripoSR
  mc_resolution?: number;
  octree_depth?: number; // TripoSG
  face_limit?: number;
  resolution?: 1024 | 1536; // Pixal3D: requested
  resolution_used?: number; // Pixal3D: what the cascade ran at (lower when the token cap was hit)
  fov?: number; // Pixal3D: requested FOV in degrees, 0 = estimated
  fov_used?: number;
  texture_size?: number;
  full_faces?: number; // faces of the undecimated mesh (model_urls.glb_full)
  engine: string;
  vertices?: number;
  faces?: number;
  front?: string; // "+z" when known to face +Z
  textured?: boolean;
  front_yaw?: number;
  uv_size?: number;
  baker?: "upstream" | "lean";
  remeshed?: boolean; // MeshFlow output
  steps?: number;
  fill_holes?: boolean;
  source_faces?: number;
  use_image?: boolean; // remesh guided by a reference image (DINOv3)
  detail?: RemeshDetail;
  guidance?: number;
  rigged?: boolean; // SkinTokens output: the GLB carries a skeleton and skin weights
  bones?: number;
  style?: RigStyle;
  beams?: number;
  temperature?: number;
  segmented?: boolean; // PartField output: one GLB node / OBJ object per part
  parts?: number;
  animated?: boolean; // UniMate output: the GLB carries an animation clip
  caption?: string; // the prompt as UniMate read it ("An object …")
  frames?: number;
  fps?: number;
  joint_labels?: string[];
  clips?: string[]; // animation names in the GLB, oldest first
}

export type RigStyle = "auto" | "humanoid";

export type RemeshDetail = "low" | "normal" | "high" | "max";

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
  model_urls?: { glb: string; obj: string; glb_full?: string }; // glb_full: undecimated TripoSG mesh
  input_url?: string;
  image_urls?: { texture?: string; views?: string };
  thumb_rendered?: boolean;
}

export type JobStatus = "queued" | "loading" | "running" | "done" | "error" | "cancelled";

export interface Job {
  id: string;
  kind: "image" | "multiview" | "model3d" | "texture" | "remesh" | "rig" | "segment" | "animate";
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

export interface TextureBaker {
  baker: "upstream" | "lean";
  note: string | null; // why it fell back to "lean", if it did
}

export interface MultiviewSupport {
  t2mv: boolean;
  i2mv: boolean;
  ig2mv: boolean;
  tg2mv: boolean;
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
    request<{
      checkpoints: Checkpoint[];
      multiview: MultiviewSupport;
      model3d: {
        triposr: boolean;
        triposg: boolean;
        pixal3d: boolean;
        pixal3d_note: string | null;
        pixal3d_mv: boolean;
        pixal3d_mv_note: string | null;
        meshflow: boolean;
        meshflow_image: boolean;
        skintokens: boolean;
        partfield: boolean;
        unimate: boolean;
      };
      texture: TextureBaker;
    }>("/api/models"),
  samplers: () => request<{ samplers: Sampler[] }>("/api/samplers"),
  generateImage: (params: Omit<ImageParams, "model_name" | "sampler_label" | "arch"> & { count: number }) =>
    request<{ jobs: Job[] }>("/api/generate/image", { method: "POST", body: JSON.stringify(params) }),
  generateMultiview: (params: MultiviewRequest) =>
    request<{ jobs: Job[] }>("/api/generate/multiview", { method: "POST", body: JSON.stringify(params) }),
  generate3d: (params: {
    source_asset?: string | null;
    source_view?: number;
    source_views?: number[]; // Pixal3D: several views of a multiview set, the front among them
    source_image?: string | null;
    remove_bg: boolean;
    engine: "triposr" | "triposg" | "pixal3d";
    foreground_ratio?: number;
    mc_resolution?: number;
    steps?: number;
    guidance?: number;
    seed?: number;
    octree_depth?: number;
    face_limit?: number;
    resolution?: number; // Pixal3D
    fov?: number;
    texture_size?: number;
  }) => request<{ jobs: Job[] }>("/api/generate/3d", { method: "POST", body: JSON.stringify(params) }),
  generateTexture: (params: {
    source_asset: string;
    model: string;
    mode: "ig2mv" | "tg2mv";
    reference_asset?: string | null;
    reference_image?: string | null;
    remove_bg: boolean;
    prompt: string;
    steps: number;
    cfg: number;
    sampler: string;
    seed: number;
    reference_scale: number;
    front_yaw: number;
    uv_size: number;
  }) => request<{ jobs: Job[] }>("/api/generate/texture", { method: "POST", body: JSON.stringify(params) }),
  generateRemesh: (params: {
    source_asset: string;
    steps: number;
    seed: number;
    count: number;
    fill_holes: boolean;
    detail: RemeshDetail;
    use_image: boolean;
    reference_asset?: string | null;
    reference_image?: string | null;
    remove_bg: boolean;
    guidance: number;
  }) =>
    request<{ jobs: Job[] }>("/api/generate/remesh", { method: "POST", body: JSON.stringify(params) }),
  generateRig: (params: { source_asset: string; style: RigStyle; seed: number; count: number; beams: number; temperature: number }) =>
    request<{ jobs: Job[] }>("/api/generate/rig", { method: "POST", body: JSON.stringify(params) }),
  generateAnimate: (params: { source_asset: string; prompt: string; seed: number; count: number; guidance: number }) =>
    request<{ jobs: Job[] }>("/api/generate/animate", { method: "POST", body: JSON.stringify(params) }),
  generateSegment: (params: { source_asset: string; parts: number; seed: number }) =>
    request<{ jobs: Job[] }>("/api/generate/segment", { method: "POST", body: JSON.stringify(params) }),
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
