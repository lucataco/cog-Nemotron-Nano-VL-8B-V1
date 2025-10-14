## Nemotron Nano VL 8B Cog Model

Multi-modal (vision + language) inference for NVIDIA's Llama-3.1 Nemotron Nano VL 8B using Replicate Cog.

This repository wraps a local model layout under `checkpoints/` (LLM + tokenizer + image processor config) and a RADIO v2 vision backbone placed under `Radiov2/`. The predictor exposes a simple single-image chat interface.

### Features
- Single-image question answering via `image` + `question` inputs
- Uses local RADIO v2 vision backbone in `Radiov2/` (no network fetches required)
- Offline mode defaults enabled for quiet logs and reproducibility
- CUDA GPU execution (`device_map="cuda"`)

## Quickstart

### Prerequisites
- NVIDIA GPU with recent CUDA drivers (tested with CUDA 12.4 container base)
- Docker installed and able to run NVIDIA CUDA images
- `cog` CLI installed (`pip install cog`)

### Project Layout
- `predict.py` — Cog predictor entrypoint
- `cog.yaml` — Cog build/runtime configuration
- `checkpoints/` — Local model config and weights for Nemotron Nano VL 8B
- `Radiov2/` — Local RADIO v2 vision backbone code + weights
- `radio/` — Thin shim to load `Radiov2` (local-only)

### Setup

Download the RADIO v2 weights:

```bash
huggingface-cli download nvidia/C-RADIOv2-H --local-dir Radiov2
```


### Running a prediction

Single image inference:

```bash
cog predict -i image=@field.jpg -i question='Describe the image.'
```

The `image` input is a single file. The `question` is a string prompt. The output is a single string response.

### Example output

```text
The image captures a serene scene of a wooden boardwalk meandering through a lush, green field... 
```

## Inputs and outputs

### Inputs
- `image` (Path, required): single image file, e.g. `@/path/to/image.jpg`
- `question` (str, optional, default: "Describe the image.")
- `max_new_tokens` (int, optional, default: 1024)
- `temperature` (float, optional, default: 0.0)

### Output
- `str`: generated response text

## Offline and local weights

This repository is configured to avoid downloading code from Hugging Face at runtime:
- `Radiov2/` contains RADIO v2 source files and `model.safetensors`
- `radio/hf_model.py` imports from `Radiov2` only (no network fallback)
- `predict.py` sets the following environment variables during setup:
  - `TRANSFORMERS_OFFLINE=1`
  - `HF_HUB_OFFLINE=1`
  - `HF_HUB_DISABLE_TELEMETRY=1`
  - `HF_HUB_DISABLE_PROGRESS_BARS=1`
  - `TRANSFORMERS_VERBOSITY=error`, `HF_HUB_VERBOSITY=error`

If `checkpoints/` does not exist at first run, the LLM bundle is fetched once via the `MODEL_URL` configured in `predict.py`. To be fully offline, ensure `checkpoints/` exists before running Cog.

## Notes on performance
- The model uses `bfloat16` where applicable
- `attn_implementation` is set to `eager` for maximum compatibility; feel free to experiment if your GPU supports flash attention
- The image processor performs dynamic tiling for high-resolution inputs

## Troubleshooting
- "GPU not found": ensure Docker has access to your NVIDIA GPU (`--gpus all`) and correct drivers are installed
- "Module import errors": verify `Radiov2/` exists and contains `hf_model.py` and related modules; `Radiov2/__init__.py` must be present
- Excessive logs or HF download notices: confirm offline env vars are set (they are by default in `predict.py`), and that `checkpoints/` + `Radiov2/` are present

## License
- RADIO v2 code and assets in `Radiov2/` are subject to their respective licenses (see files and headers)
- Nemotron Nano VL configuration here is adapted for inference; please review upstream licenses and usage terms
- This wrapper code is provided under the same license as this repository


