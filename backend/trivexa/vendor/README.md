# Vendored third-party code

Copied (not pip-installed) to avoid each project's heavy, unrelated dependencies.
Only the files needed for inference are included; local edits are marked `# Trivexa:`.

| Package | Source | Commit | License |
|---|---|---|---|
| `mvadapter/` | https://github.com/huanngzh/MV-Adapter | `4277e0018232bac82bb2c103caf0893cedb711be` | Apache-2.0 (`mvadapter/LICENSE`) |
| `birefnet/` | https://huggingface.co/ZhengPeng7/BiRefNet | `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` | MIT |

Weights are not vendored: MV-Adapter adapters are downloaded to `data/models/mv-adapter/`,
BiRefNet weights are read from `<models dir>/background_removal/birefnet.safetensors`.
