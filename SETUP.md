# ============================================================
#  DataChat — Complete Setup Guide
#  RTX 3050 6GB  |  Ollama + HuggingFace  |  CSV/PDF/XLSX
# ============================================================


## STEP 1 — Install Python 3.10+
# Download from https://www.python.org/downloads/


## STEP 2 — Create virtual environment
python -m venv venv

# Activate:
# Windows:   venv\Scripts\activate
# Mac/Linux: source venv/bin/activate


## STEP 3 — Install PyTorch with CUDA (GPU support for RTX 3050)
# Check your CUDA version first: nvidia-smi (top right corner)
# Then install the matching version:

# For CUDA 12.1 (most common for RTX 3050 with recent drivers):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older drivers):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU works:
# python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Should print: True  NVIDIA GeForce RTX 3050 Laptop GPU


## STEP 4 — Install remaining dependencies
pip install -r requirements.txt


## STEP 5 — Install Ollama
# Download from: https://ollama.com/download
# Install it (Windows installer or brew install ollama on Mac)


## STEP 6 — Pull a model (choose based on your use case)
# For RTX 3050 6GB VRAM, recommended models:

ollama pull llama3.2        # 2GB VRAM — good quality, fast  ✅ RECOMMENDED
ollama pull gemma2:2b       # 1.5GB VRAM — fastest
ollama pull mistral         # 4GB VRAM — better quality
# DO NOT use 13B+ models — they won't fit in 6GB VRAM


## STEP 7 — Start everything

# Terminal 1 — Start Ollama:
ollama serve

# Terminal 2 — Start Flask:
python app.py

# Open browser:
# http://localhost:5000


## ── GPU vs CPU — What runs where ───────────────────────────
#
#  Component            | RTX 3050 (with setup above)
#  ---------------------|------------------------------------
#  Embeddings (MiniLM)  | ✅ CUDA — very fast
#  Ollama LLM           | ✅ CUDA — num_gpu=99 offloads all layers
#  HuggingFace LLM      | ✅ CUDA — float16 (half precision)
#  Pandas analytics     | CPU — (doesn't need GPU)
#  FAISS search         | CPU — (fast enough, ~1ms per query)
#
#  With GPU: responses in 2-5 seconds
#  Without GPU (CPU only): responses in 30-120 seconds


## ── Upgrade HuggingFace model (optional) ───────────────────
# In rag_engine.py, change model_id on line ~180:
#
# model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'   # 0.6GB VRAM (default)
# model_id = 'microsoft/phi-2'                        # 2.5GB VRAM (better)
# model_id = 'mistralai/Mistral-7B-Instruct-v0.2'    # 5.0GB VRAM (best)


## ── Troubleshooting ─────────────────────────────────────────
#
# "No module named torch"          → redo Step 3
# "CUDA not available"             → install CUDA toolkit from nvidia.com
#                                    or reinstall torch with correct cu version
# "Ollama timed out"               → model loading into VRAM, wait 30s, retry
# "No models installed"            → run: ollama pull llama3.2
# "500 error on upload"            → check Flask terminal for Python traceback
# "Badge still shows No dataset"   → upload returned error, check terminal
