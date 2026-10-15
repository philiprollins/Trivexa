import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# Trivexa: NATTEN's na2d (a compiled CUDA extension) replaced by plain PyTorch, and the legacy return_weights
# path dropped. Upstream resizes keys/values to the query size (nearest-exact) and runs dilated neighborhood
# attention with dilation = query size // key size. With an integer ratio, every dilation group of that resized
# grid is exactly the low-res grid again, so each query at (y, x) attends to the kernel x kernel low-res window
# around (y // d, x // d), shifted to stay inside the grid, as NATTEN does at borders. That is computed here
# directly on the low-res keys/values, without materialising them at full resolution.
def _window_starts(n, kernel_size, device):
    return (torch.arange(n, device=device) - kernel_size // 2).clamp(0, n - kernel_size)


def neighborhood_cross_attention(q, k, v, kernel_size, scale, rows_per_chunk=8):
    """q: [B, H, W, N, C] queries; k: [B, h, w, N, C], v: [B, h, w, N, Cv] at a resolution dividing H, W."""
    B, H, W, N, _ = q.shape
    h, w = k.shape[1:3]
    kh, kw = kernel_size
    if H % h or W % w:
        raise ValueError(f"query size {H}x{W} is not a multiple of key size {h}x{w}")
    if h < kh or w < kw:
        raise ValueError(f"key grid {h}x{w} is smaller than the {kh}x{kw} neighborhood")
    dy, dx = H // h, W // w
    ys = _window_starts(h, kh, q.device)[:, None] + torch.arange(kh, device=q.device)  # [h, kh]
    xs = _window_starts(w, kw, q.device)[:, None] + torch.arange(kw, device=q.device)  # [w, kw]
    out = q.new_empty(B, H, W, N, v.shape[-1])
    for r0 in range(0, h, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, h)
        rows = ys[r0:r1]  # [r, kh]
        # Neighborhoods of the low-res cells in this chunk: [B, r, w, kh*kw, N, C]
        kn = k[:, rows][:, :, :, xs].permute(0, 1, 3, 2, 4, 5, 6).flatten(3, 4)
        vn = v[:, rows][:, :, :, xs].permute(0, 1, 3, 2, 4, 5, 6).flatten(3, 4)
        qc = q[:, r0 * dy : r1 * dy].reshape(B, r1 - r0, dy, w, dx, N, -1)
        attn = torch.einsum("biajcnd,bijknd->biajcnk", qc, kn) * scale
        attn = attn.softmax(dim=-1)
        oc = torch.einsum("biajcnk,bijknd->biajcnd", attn, vn)
        out[:, r0 * dy : r1 * dy] = oc.reshape(B, (r1 - r0) * dy, W, N, -1)
    return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        kernel_size=(9, 9),
        **kwargs,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.kernel_size = kernel_size

        self.scale = (dim // num_heads) ** -0.5

    def forward(self, q, k, v, image=None, **kwargs):
        hq, wq = q.shape[-2:]
        hk, wk = k.shape[-2:]
        self.dilation = (hq // hk, wq // wk)

        q = rearrange(q, "b (n d) h w -> b h w n d", n=self.num_heads)
        k = rearrange(k, "b (n d) h w -> b h w n d", n=self.num_heads).to(q.dtype)
        v = rearrange(v, "b (n d) h w -> b h w n d", n=self.num_heads).to(q.dtype)
        out = neighborhood_cross_attention(q, k, v, self.kernel_size, self.scale)
        return rearrange(out, "b h w n d -> b (n d) h w")
