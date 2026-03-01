# MedGemma 27B Server

Medical AI assistant powered by Google's MedGemma 27B model, running on DGX Spark.

**Architecture:** vLLM serves the model as an OpenAI-compatible API (port 8000), and Gradio provides a web chat UI with image upload (port 7860).

## Prerequisites

- DGX Spark (NVIDIA GB10, 128GB unified memory)
- CUDA 13.0+ / sm_121 (Blackwell)
- Hugging Face account with access to `google/medgemma-27b-it`
- Python 3.10+

## Setup

### 1. Configure your HF token

Edit `.env` and set your `HF_TOKEN`:

```bash
cp .env .env.backup  # optional
nano .env
```

Also export it for model download:
```bash
export HF_TOKEN=<your-token>
huggingface-cli login --token $HF_TOKEN
```

### 2. Create virtual environment and install vLLM

```bash
python3 -m venv .venv
source .venv/bin/activate

# DGX Spark requires cu130 wheels
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install vllm --extra-index-url https://wheels.vllm.ai/nightly/cu130

# Install UI dependencies
pip install -r requirements.txt
```

> **Note:** Do NOT install `flash-attn` separately — vLLM bundles FlashInfer.

### 3. Start services

```bash
# Start everything
./start_all.sh

# Or start individually
./start_vllm.sh   # API server only
./start_ui.sh     # Web UI only (requires vLLM running)
```

### 4. Stop services

```bash
./stop_all.sh
```

## Usage

### Web UI

Open http://localhost:7860 in your browser. You can:
- Ask medical questions via text
- Upload medical images (X-rays, pathology, dermatology) for analysis
- Select different assistant modes (General, Radiologist, Dermatology, Pathology)

### API

The vLLM server exposes an OpenAI-compatible API on port 8000:

```bash
# Text query
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/medgemma-27b-it",
    "messages": [{"role": "user", "content": "What are the symptoms of pneumonia?"}]
  }'

# Python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="google/medgemma-27b-it",
    messages=[{"role": "user", "content": "What are the symptoms of pneumonia?"}]
)
print(response.choices[0].message.content)
```

## Logs

- vLLM server: `vllm.log`
- Gradio UI: `gradio.log`

## DGX Spark Notes

- GB10 GPU uses sm_121 (Blackwell) — standard vLLM wheels won't work
- Must use cu130-specific wheels (see setup above)
- 27B model in bfloat16 needs ~54GB VRAM — fits comfortably in 128GB unified memory
- Set `TORCH_CUDA_ARCH_LIST=12.1a` if building from source

## Disclaimer

This is an AI research tool. Do not use for clinical decision-making. Always consult qualified healthcare professionals for medical advice.
