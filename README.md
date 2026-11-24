---
title: 3-Layer Guardrailed Hybrid KYC/AML Compliance Agent
emoji: 🛡
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
app_port: 7860
pinned: false
---

# 🛡3-Layer Guardrailed Hybrid KYC/AML Compliance Agent

An autonomous yet deterministically guardrailed **Anti-Money Laundering (AML) & Know Your Customer (KYC)** transaction triage system tailored for Hong Kong banking corridors (SWIFT, CHATS/RTGS, and FPS).

Built with **LangGraph**, **NiceGUI**, **Pydantic**, and a self-contained local **Qwen2.5-1.5B-Instruct (GGUF)** model via `llama-cpp-python`—requiring zero external paid LLM APIs or runtime dataset downloads.

---

## 🏗3-Layer Hybrid Architecture

1. **Layer 1 Mandatory Statutory Compliance Gate (Deterministic Python)**
   * **UN / OFAC / EU Sanctions Screening:** Screens both remitting (`From Bank`) and beneficiary (`To Bank`) institutions against **73,930+ consolidated sanctioned entities** from OpenSanctions (`compliance_data.zip`).
   * **FATF High-Risk Jurisdiction Check (Rec. 19):** Flags payment corridors involving FATF high-risk or monitored jurisdictions (`Iran`, `Myanmar`, `Vietnam`, `Syria`, `Russia`, `North Korea`).
   * **HKMA AMLO CDD Threshold:** Flags transactions exceeding **HKD 120,000** under Hong Kong Cap. 615.
   * **Dormant Account Reactivation Alert:** Detects movement exceeding **HKD 50,000** on inactive/dormant accounts.

2. **Layer 2 Autonomous AI Detective Agent (LangGraph ReAct Tool Loop)**
   * Dynamically evaluates Layer 1 findings and account profiles.
   * Automatically invokes `check_enhanced_cdd_requirements` (HKMA AML/CFT Guideline Section 4) and `search_kyc_regulation` (200 FinCEN 31 CFR 1010/1020 Customer Due Diligence clauses) when red flags or dormant/corporate accounts are present—or short-circuits clean active transfers to save latency.

3. **Layer 3 Guardrail Synthesizer & Hard Compliance Overrides**
   * Compiles structured violation reports (`Pydantic`) with interactive bookmark navigation linking every detected violation directly to its underlying statutory or watchlist evidence.
   * Enforces deterministic risk floors (e.g., any `UN-OFAC-SANCTIONS` hit forces `CRITICAL` risk regardless of LLM variance).

---

## 📦 Repository Structure

```text
├── app.py                 # Main LangGraph 3-layer agent & NiceGUI web interface
├── compliance_data.zip    # Pre-packaged OpenSanctions CSV (73,930 entities) + FinCEN CDD text
├── Dockerfile             # Pre-bakes Qwen2.5-1.5B-Instruct GGUF & exposes port 8085
├── requirements.txt       # Python dependencies (NiceGUI, LangGraph, llama-cpp-python)
├── .gitignore             # Excludes local model cache & virtual environments
└── README.md              # Project documentation & Hugging Face Space metadata
```text
└── README.md              # Project documentation & Hugging Face Space metadata
```

---

## 🚀 Running Locally

### Option A: Native Python Execution (Quick Start)

Run directly with Python—model downloads automatically on first launch into `hf_model_cache/`:

```bash
# Install dependencies (Windows may need --extra-index-url for CPU builds)
pip install -r requirements.txt --extra-index-url https://github.io

# Run the app — port 8085 by default on Windows/Linux/macOS
python app.py
```

Open `http://localhost:8085` in your browser. On first launch, **Qwen2.5-1.5B-Instruct-GGUF** (~986 MB) downloads automatically from HuggingFace and runs completely offline thereafter.</parameter>
<parameter=old_text>
```text
└── README.md              # Project documentation & Hugging Face Space metadata
```

## 🚀 Running Locally

### Option A: Native Python Execution (Quick Start)

Run directly with Python—model downloads automatically on first launch into `hf_model_cache/`:

```bash
# Install dependencies (Windows may need --extra-index-url for CPU builds)
pip install -r requirements.txt --extra-index-url https://github.io

# Run the app — port 8085 by default on Windows/Linux/macOS
python app.py
```

Open `http://localhost:8085` in your browser. On first launch, **Qwen2.5-1.5B-Instruct-GGUF** (~986 MB) downloads automatically from HuggingFace and runs completely offline thereafter.</parameter>
<parameter=start_line>
48
```

## 🚀 Running Locally

### Option A: Native Python Execution (Quick Start)

Run directly with Python—model downloads automatically on first launch into `hf_model_cache/`:

```bash
# Install dependencies (Windows may need --extra-index-url for CPU builds)
pip install -r requirements.txt --extra-index-url https://github.io

# Run the app — port 8085 by default on Windows/Linux/macOS
python app.py
```

Open `http://localhost:8085` in your browser. On first launch, **Qwen2.5-1.5B-Instruct-GGUF** (~986 MB) downloads automatically from HuggingFace and runs completely offline thereafter.

### Option B: Dockerized Deployment (Production-Ready)

Docker pre-bakes the model cache for instant startup anywhere. No `pip install` required on host!

```bash
# Build & run in one command — maps port 8085 from container to host
docker run --name kyc-aml-agent -p 8085:8085 -d --restart unless-stopped \
  ghcr.io/yourusername/kyc-aml-agent

# Open the interface  
open http://localhost:8085
```

**Docker Volume Mounts (Optional):** For persistent model cache updates without rebuilding:

```bash
docker run --name kyc-aml-agent -p 8085:8085 \
  -v /path/to/local/cache:/app/hf_model_cache \
  -d ghcr.io/yourusername/kyc-aml-agent
```</parameter>
<parameter=old_text>
### Option B: Dockerized Deployment (Production-Ready)

Docker pre-bakes the model cache for instant startup anywhere. No `pip install` required on host!

```bash
# Build & run in one command — maps port 8085 from container to host
docker run --name kyc-aml-agent -p 8085:8085 -d --restart unless-stopped \
  ghcr.io/yourusername/kyc-aml-agent

# Open the interface  
open http://localhost:8085
```

**Docker Volume Mounts (Optional):** For persistent model cache updates without rebuilding:

```bash
docker run --name kyc-aml-agent -p 8085:8085 \
  -v /path/to/local/cache:/app/hf_model_cache \
  -d ghcr.io/yourusername/kyc-aml-agent
```</parameter>
<parameter=start_line>
64
Open `http://localhost:8085` in your browser. On first launch, **Qwen2.5-1.5B-Instruct-GGUF** (~986 MB) downloads automatically from HuggingFace and runs completely offline thereafter.</parameter>
<parameter=old_text>
## 🚀 Local Execution (Windows / Linux / macOS)</parameter>
<parameter=start_line>
52
---

## 🚀 Local Execution (Windows / Linux / macOS)

```bash
pip install -r requirements.txt --extra-index-url https://github.io
python app.py
```
Open `http://localhost:8085` in your browser. On first launch, `Qwen2.5-1.5B-Instruct-GGUF` (~986 MB) downloads automatically into `hf_model_cache/` and runs completely offline thereafter.
