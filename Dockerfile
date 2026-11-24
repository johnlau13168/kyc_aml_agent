FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# libgomp1 is required by the llama-cpp-python CPU wheel (OpenMP runtime).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-bake the quantised model so the container starts offline.
# Build with --build-arg PRELOAD_MODEL=0 to download it on first run instead.
ARG PRELOAD_MODEL=1
RUN if [ "$PRELOAD_MODEL" = "1" ]; then \
        python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('Qwen/Qwen2.5-3B-Instruct-GGUF', \
'qwen2.5-3b-instruct-q4_k_m.gguf', cache_dir='/app/hf_model_cache')"; \
    fi

COPY app.py kyc_aml_compliance.agent.py compliance_data.zip ./

# SPACE_ID selects port 7860 and CPU-only inference (no GPU layers).
ENV SPACE_ID=docker \
    HOME=/app

EXPOSE 7860

CMD ["python", "app.py"]
