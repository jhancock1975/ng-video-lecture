#!/usr/bin/env bash
set -euo pipefail

### --- Configurable bits ---
MODEL_ID="openai/gpt-oss-120b"
MODEL_DIR="/root/models/gpt-oss-120b"
SERVE_PORT="8000"
API_KEY="demo-key"
SERVED_NAME="openai/gpt-oss-120b"   # this is the name your clients will use in "model": ...
PYTHON_VER="3.11"
### --------------------------

echo "[1/9] Apt setup & Python ${PYTHON_VER}"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y software-properties-common curl ca-certificates gnupg lsb-release || true
sudo add-apt-repository -y ppa:deadsnakes/ppa || true
sudo apt-get update -y || true
sudo apt-get install -y \
  "python${PYTHON_VER}" "python${PYTHON_VER}-venv" "python${PYTHON_VER}-dev" \
  jq vim || true

echo "[2/9] Install uv (Python package manager)"
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure uv is on PATH for this session
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "[3/9] Create & activate virtualenv"
uv venv --python "python${PYTHON_VER}" "$HOME/ven"
# shellcheck disable=SC1091
source "$HOME/ven/bin/activate"


echo "[4/9] Install PyTorch (CUDA 12.1 wheels)"
uv pip install --upgrade pip setuptools wheel
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[5/9] Verify CUDA with PyTorch"
python - <<'PY'
import torch, sys
assert torch.cuda.is_available(), "CUDA not available (torch.cuda.is_available() == False)"
print("✓ PyTorch sees CUDA:", torch.version.cuda, " Device(s):", torch.cuda.device_count())
PY

echo "[6/9] Install vLLM + Hugging Face tools"
uv pip install "vllm>=0.6.2" "huggingface_hub>=0.24"  # provides 'hf' CLI

echo "[7/9] Download model weights (no original/**, no *.bin/*.pt)"
mkdir -p "$MODEL_DIR"
# Prefer the modern 'hf' CLI. Fallback to 'huggingface-cli' if needed.
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --exclude "original/**" "*.bin" "*.pt"
else
  echo "Note: 'hf' not found; using deprecated huggingface-cli"
  uv pip install "huggingface_hub[cli]"
  huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --exclude "original/**" "*.bin" "*.pt"
fi

echo "[8/9] Launch vLLM server (OpenAI-compatible API)"
# Tip: FA2 is usually the safe bet on A100; FA3 is Hopper-oriented.
export HF_HUB_OFFLINE=1
export VLLM_FLASH_ATTN_VERSION=2

# Tune these if you still hit KV-cache errors:
MAX_MODEL_LEN=131072
MAX_BATCHED_TOKENS=1024
MAX_NUM_SEQS=16

# Start server in the foreground (remove 'exec' if you prefer to nohup it)
exec vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port "$SERVE_PORT" \
  --api-key "$API_KEY" \
  --served-model-name "$SERVED_NAME" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --no-enable-prefix-caching

# [9/9] (Optional) In another shell, you can verify:
#   curl -s http://localhost:${SERVE_PORT}/v1/models -H "Authorization: Bearer ${API_KEY}" | jq
# and call chat:
#   curl -s http://localhost:${SERVE_PORT}/v1/chat/completions \
#     -H "Authorization: Bearer ${API_KEY}" -H "Content-Type: application/json" \
#     -d "{\"model\":\"${SERVED_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":32}" | jq
