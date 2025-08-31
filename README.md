# originally forked from Karpathy nanogpt-lecture
## notes for installing vllm

What has worked so far for installing vllm in Vast.ai, use A100 instance with 200 GB storage:

```
add-apt-repository -y ppa:deadsnakes/ppa
# install python3.11
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# make a virtual environment
uv venv ~/ven
source ~/ven/bin/activate

uv pip install --upgrade pip setuptools wheel jq vim
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# confirm pytorch
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('✓ PyTorch CUDA working')"
uv pip install vllm huggingface_hub
# downloads weights on first run and serves OpenAI-compatible API on :8000

# 1) Pick a persistent place
export MODEL_DIR=/root/models/gpt-oss-120b

# 2) Download ONLY what vLLM needs (no original/**, no *.bin/*.pt)
huggingface-cli download openai/gpt-oss-120b \
  --local-dir "$MODEL_DIR" \
  --local-dir-use-symlinks False \
  --exclude "original/**" "*.bin" "*.pt"

# 3) Run vLLM strictly from disk (no network pulls)
export HF_HUB_OFFLINE=1
python -m vllm.entrypoints.api_server \
  --model "$MODEL_DIR"

# had to add --gpu-mem... to avoid cache memory insufficient error

HF_HUB_OFFLINE=1 \
vllm serve "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key demo-key \
  --gpu-memory-utilization 0.95
```
It all worked.  I can expose the port by configuring docker options in the Vast.ai template for the instance running, but that port gets mapped to some random port.  
In the Vast.ai console, I have to click on the IP address of the instance to get the port mapping. See the image below.

<img width="437" height="429" alt="image" src="https://github.com/user-attachments/assets/9dc70053-9ad2-4c97-b775-5728df1d9ea8" />



# nanogpt-lecture

Code created in the [Neural Networks: Zero To Hero](https://karpathy.ai/zero-to-hero.html) video lecture series, specifically on the first lecture on nanoGPT. Publishing here as a Github repo so people can easily hack it, walk through the `git log` history of it, etc.

NOTE: sadly I did not go too much into model initialization in the video lecture, but it is quite important for good performance. The current code will train and work fine, but its convergence is slower because it starts off in a not great spot in the weight space. Please see [nanoGPT model.py](https://github.com/karpathy/nanoGPT/blob/master/model.py) for `# init all weights` comment, and especially how it calls the `_init_weights` function. Even more sadly, the code in this repo is a bit different in how it names and stores the various modules, so it's not possible to directly copy paste this code here. My current plan is to publish a supplementary video lecture and cover these parts, then I will also push the exact code changes to this repo. For now I'm keeping it as is so it is almost exactly what we actually covered in the video.

### License

MIT
