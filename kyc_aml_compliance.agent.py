import asyncio
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from typing import Annotated, TypedDict, List, Literal
from pydantic import BaseModel, Field
from nicegui import ui, run
#from langchain_openai import ChatOpenAI
#  from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AnyMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from huggingface_hub import hf_hub_download
from langchain_community.chat_models import ChatLlamaCpp

if sys.platform == "win32":
    from asyncio import proactor_events
    _orig = proactor_events._ProactorBasePipeTransport._call_connection_lost
    def _quiet(self, exc):
        try:
            _orig(self, exc)
        except ConnectionResetError:
            pass
    proactor_events._ProactorBasePipeTransport._call_connection_lost = _quiet


def log_debug(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# =====================================================================
# 1. LOAD PRE-DOWNLOADED KYC & OPENSANCTIONS FROM SINGLE ZIP ARCHIVE
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ZIP_PATH = os.path.join(BASE_DIR, "compliance_data.zip")


def load_kyc_from_zip() -> list[Document]:
    full_text = ""
    if os.path.exists(DATA_ZIP_PATH):
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
            txt_files = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if txt_files:
                target_txt = txt_files[0]
                log_debug(f"[ZIP KYC] Reading '{target_txt}' from {DATA_ZIP_PATH}...")
                full_text = zf.read(target_txt).decode("utf-8", errors="ignore")
    else:
        log_debug(f"[ZIP KYC] Warning: Archive not found at {DATA_ZIP_PATH}")

    if not full_text:
        log_debug("[ZIP KYC] Using built-in minimal FinCEN fallback text.")
        full_text = (
            "Customer Due Diligence (CDD) Requirements for Financial Institutions (31 CFR 1010.230): "
            "Covered financial institutions must identify and verify the identity of beneficial owners "
            "of legal entity customers (each individual owning 25 percent or more equity interest and "
            "one individual with significant responsibility to control, manage, or direct the legal entity), "
            "understand the nature and purpose of customer relationships to develop a customer risk profile, "
            "and conduct ongoing monitoring to identify and report suspicious transactions."
        )

    chunks = [full_text[i:i + 600] for i in range(0, min(len(full_text), 120000), 600)]
    docs = []
    for idx, chunk in enumerate(chunks):
        text_piece = chunk.strip()
        if len(text_piece) not in range(0, 101):
            docs.append(
                Document(
                    page_content=text_piece,
                    metadata={"source": "FinCEN CDD Rule (31 CFR 1010/1020)", "chunk_id": f"KYC-SEC-{idx + 1}"}
                )
            )
    log_debug(f"[ZIP KYC] Loaded {len(docs)} FinCEN regulatory chunks into memory.")
    return docs


def load_sanctions_from_zip() -> list[dict]:
    records = []
    if os.path.exists(DATA_ZIP_PATH):
        with zipfile.ZipFile(DATA_ZIP_PATH, "r") as zf:
            csv_files = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if csv_files:
                target_csv = csv_files[0]
                log_debug(f"[ZIP WATCHLIST] Streaming '{target_csv}' from {DATA_ZIP_PATH}...")
                with zf.open(target_csv, "r") as raw_stream:
                    text_stream = io.TextIOWrapper(raw_stream, encoding="utf-8", errors="ignore")
                    for row in csv.DictReader(text_stream):
                        records.append({
                            "id": row.get("id", ""),
                            "schema": row.get("schema", ""),
                            "name": row.get("name", ""),
                            "countries": row.get("countries", "")
                        })
    else:
        log_debug(f"[ZIP WATCHLIST] Warning: Archive not found at {DATA_ZIP_PATH}")

    log_debug(f"[ZIP WATCHLIST] Loaded ALL {len(records):,} sanctioned entities from zip.")
    return records


KYC_DOCS = load_kyc_from_zip()
SANCTIONS_DB = load_sanctions_from_zip()


# # =====================================================================
# # 2. HYBRID CLOUD (HUGGING FACE FREE API) / LOCAL (OLLAMA) LLM FACTORY
# # =====================================================================
# def get_hybrid_llm():
#     hf_token = os.getenv("HF_TOKEN") or os.getenv("LLM_API_KEY")
#     if hf_token:
#         hf_router_url = "https://router." + "huggingface.co/v1"
#         return ChatOpenAI(
#             api_key=hf_token,
#             base_url=os.getenv("LLM_BASE_URL", hf_router_url),
#             model=os.getenv("LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
#             temperature=0.0,
#             max_tokens=800
#         )
#     return ChatOllama(
#         model="llama3.2:latest",
#         base_url="http://127.0.0.1:11434",
#         temperature=0.0,
#         num_predict=800
#     )
#
#
#
#
# llm = get_hybrid_llm()


# =====================================================================
# 2. HYBRID CLOUD (HUGGING FACE FREE API) / LOCAL (OLLAMA) LLM FACTORY
# =====================================================================
# =====================================================================
# 2. HUGGING FACE SERVERLESS INFERENCE LLM FACTORY (HUGGING FACE ONLY)
# =====================================================================
# =====================================================================
# 2. SELF-CONTAINED LOCAL GGUF LLM (ZERO API KEYS / ZERO 402 ERRORS)
# =====================================================================
def get_local_gguf_llm():
    repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    log_debug(f"[LLM INIT] Resolving local GGUF model: {repo_id} ({filename})...")

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=os.path.join(BASE_DIR, "hf_model_cache")
    )

    is_hf_space = os.getenv("SPACE_ID") is not None
    cpu_threads = max(2, min(os.cpu_count() or 2, 4))
    gpu_layers = 0 if is_hf_space else -1

    log_debug(f"[LLM INIT] Loading Qwen2.5-3B-Instruct (gpu_layers={gpu_layers}, threads={cpu_threads})...")

    llm_instance = ChatLlamaCpp(
        model_path=model_path,
        chat_format="chatml-function-calling",
        n_ctx=2048,
        n_threads=cpu_threads,
        n_batch=512,
        n_gpu_layers=gpu_layers,
        temperature=0.0,
        max_tokens=500,
        verbose=False
    )
    return llm_instance, "Local GGUF: Qwen2.5-3B-Instruct"


llm, ACTIVE_LLM_LABEL = get_local_gguf_llm()





ORIGIN_OPTIONS = [
    "Hong Kong", "Singapore", "United Kingdom", "United States", "Japan",
    "Mainland China", "Iran", "Myanmar", "Vietnam", "Syria"
]


# =====================================================================
# 3. LAYER 2 AUTONOMOUS AGENT TOOLS (Legal & KYC Deep-Dive)
# =====================================================================
@tool
def search_kyc_regulation(topic_query: str) -> str:
    """Searches the pre-loaded FinCEN/international KYC Customer Due Diligence regulatory document for legal clauses."""
    q_words = set(re.findall(r"[a-z]{4,}", topic_query.lower()))
    scored = []
    for doc in KYC_DOCS:
        d_words = set(re.findall(r"[a-z]{4,}", doc.page_content.lower()))
        score = len(q_words & d_words)
        if score not in range(0, 1):
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for _, doc in scored[:2]]
    return json.dumps([
        {
            "ref_id": d.metadata["chunk_id"],
            "source": d.metadata["source"],
            "text": d.page_content
        }
        for d in top_docs
    ])


@tool
def check_enhanced_cdd_requirements(customer_type: str, is_dormant: bool) -> str:
    """Checks HKMA AML/CFT Guideline Section 4 for specific Enhanced Due Diligence (EDD) documentation requirements."""
    reqs = []
    if "corporate" in customer_type.lower():
        reqs.append("Ultimate Beneficial Owner (UBO >= 25% shareholding) registry & Certificate of Incumbency required.")
    if is_dormant:
        reqs.append("Independent client callback verification & fresh source-of-funds declaration required before lifting account freeze.")
    if not reqs:
        reqs.append("Standard customer identity & address verification suffices under simplified CDD.")
    return json.dumps({
        "ref_id": "HKMA-EDD-SEC4",
        "source": "HKMA Guideline on Anti-Money Laundering (Section 4)",
        "requirements": reqs
    })


LAYER2_TOOLS = [search_kyc_regulation, check_enhanced_cdd_requirements]
llm_with_tools = llm.bind_tools(LAYER2_TOOLS)


# =====================================================================
# 4. GRAPH STATE & 3-LAYER HYBRID ARCHITECTURE NODES
# =====================================================================
class ViolationItem(BaseModel):
    violation_id: str = Field(description="Sequential bookmark ID: 'VIO-1', 'VIO-2', etc.")
    title: str = Field(description="Short headline of the violation")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Severity rating")
    reference_id: str = Field(description="Exact ref_id from Layer 1 or Layer 2 (e.g., 'UN-OFAC-SANCTIONS', 'FATF-R19', 'HKMA-AMLO-120K', 'HKMA-DORMANT-50K', 'HKMA-EDD-SEC4', 'KYC-SEC-1')")
    explanation: str = Field(description="Concise explanation grounded in Layer 1 and Layer 2 findings")


class RiskReport(BaseModel):
    overall_risk: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(description="Overall risk tier")
    violations: List[ViolationItem] = Field(description="Violations found. Empty [] if clean.")
    recommended_action: str = Field(description="Operational action for compliance officer")


class HybridAgentState(TypedDict):
    transaction: dict
    layer1_checks: List[dict]
    references_map: dict
    messages: Annotated[List[AnyMessage], add_messages]
    layer2_trace: List[dict]
    report: dict
    override_applied: bool

FATF_HIGH_RISK = {"iran", "myanmar", "vietnam", "syria", "russia", "north korea"}


def layer1_mandatory_gate_node(state: HybridAgentState) -> dict:
    """LAYER 1: Deterministic Python Compliance Gate (100% guaranteed execution, zero LLM skipping)."""
    log_debug("LAYER 1 [Mandatory Gate]: Running deterministic statutory checks...")
    tx = state["transaction"]
    checks = []
    refs_map = {}

    # Check 1: Screen BOTH From Bank and To Bank against OpenSanctions
    hits = []
    for role, bank_name in [("From Bank", tx["from_bank"]), ("To Bank", tx["to_bank"])]:
        b_lower = bank_name.strip().lower()
        if b_lower and len(b_lower) not in range(0, 4):
            matches = [item for item in SANCTIONS_DB if b_lower in item["name"].lower()][:2]
            for m in matches:
                hits.append(f"{role}: {m['name']} ({m['id']})")

    if hits:
        checks.append({
            "rule": "UN / OFAC / EU Sanctions Screening (Sender & Beneficiary)",
            "ref_id": "UN-OFAC-SANCTIONS",
            "triggered": True,
            "severity": "CRITICAL",
            "detail": f"Watchlist match found -> {', '.join(hits)}"
        })
        refs_map["UN-OFAC-SANCTIONS"] = {
            "title": "International Sanctions & Watchlist Match",
            "source": "OpenSanctions Consolidated Feed (UN / OFAC / EU)",
            "text": f"Matched sanctioned entities: {', '.join(hits)}"
        }
    else:
        checks.append({
            "rule": "UN / OFAC / EU Sanctions Screening (Sender & Beneficiary)",
            "ref_id": "UN-OFAC-SANCTIONS",
            "triggered": False,
            "severity": "PASS",
            "detail": f"No watchlist hits for '{tx['from_bank']}' or '{tx['to_bank']}' across {len(SANCTIONS_DB):,} entities."
        })

    # Check 2: FATF High-Risk Jurisdiction in Either Account
    combined_accts = f"{tx['from_account']} {tx['to_account']}".lower()
    matched_fatf = [c.title() for c in FATF_HIGH_RISK if c in combined_accts]
    if matched_fatf:
        checks.append({
            "rule": "FATF High-Risk Jurisdiction (Rec. 19)",
            "ref_id": "FATF-R19",
            "triggered": True,
            "severity": "HIGH",
            "detail": f"High-risk jurisdiction detected in payment corridor: {', '.join(matched_fatf)}."
        })
        refs_map["FATF-R19"] = {
            "title": "FATF Recommendation 19 — Higher-Risk Jurisdictions",
            "source": "FATF 40 Recommendations / HKMA AMLO",
            "text": f"Corridor involving {', '.join(matched_fatf)} requires mandatory Enhanced Due Diligence (EDD)."
        }
    else:
        checks.append({
            "rule": "FATF High-Risk Jurisdiction (Rec. 19)",
            "ref_id": "FATF-R19",
            "triggered": False,
            "severity": "PASS",
            "detail": "Neither From Account nor To Account involves a FATF High-Risk jurisdiction."
        })

    # Check 3: HKMA AMLO HKD 120,000 CDD Threshold
    if tx["amount_hkd"] not in range(0, 120001):
        checks.append({
            "rule": "HKMA AMLO Threshold (Exceeds HKD 120,000)",
            "ref_id": "HKMA-AMLO-120K",
            "triggered": True,
            "severity": "HIGH",
            "detail": f"Amount HKD {tx['amount_hkd']:,} exceeds statutory HKD 120,000 CDD limit."
        })
        refs_map["HKMA-AMLO-120K"] = {
            "title": "HKMA AMLO Threshold (Exceeds HKD 120,000 CDD Trigger)",
            "source": "Hong Kong Cap. 615 Anti-Money Laundering Ordinance",
            "text": f"Transaction amount HKD {tx['amount_hkd']:,} exceeds the HKD 120,000 Customer Due Diligence threshold."
        }
    else:
        checks.append({
            "rule": "HKMA AMLO Threshold (Exceeds HKD 120,000)",
            "ref_id": "HKMA-AMLO-120K",
            "triggered": False,
            "severity": "PASS",
            "detail": f"Amount HKD {tx['amount_hkd']:,} is within standard HKD 120,000 threshold."
        })

    # Check 4: Dormant Account Alert (Either Sender or Receiver > HKD 50,000)
    is_dormant = "dormant" in combined_accts
    if is_dormant and tx["amount_hkd"] not in range(0, 50001):
        checks.append({
            "rule": "Dormant Account Alert (Exceeds HKD 50,000)",
            "ref_id": "HKMA-DORMANT-50K",
            "triggered": True,
            "severity": "HIGH",
            "detail": f"Dormant account movement ({tx['from_account']} -> {tx['to_account']}) of HKD {tx['amount_hkd']:,}."
        })
        refs_map["HKMA-DORMANT-50K"] = {
            "title": "Dormant Account Reactivation (Exceeds HKD 50,000)",
            "source": "HKMA AML/CFT Transaction Monitoring Guidelines",
            "text": f"Dormant account activity ({tx['from_account']} -> {tx['to_account']}) of HKD {tx['amount_hkd']:,} exceeds the HKD 50,000 limit."
        }
    else:
        checks.append({
            "rule": "Dormant Account Alert (Exceeds HKD 50,000)",
            "ref_id": "HKMA-DORMANT-50K",
            "triggered": False,
            "severity": "PASS",
            "detail": "No dormant account threshold breach."
        })

    triggered_count = sum(1 for c in checks if c["triggered"])
    log_debug(f"LAYER 1 COMPLETE: {triggered_count}/{len(checks)} mandatory rules triggered.")

    handoff_msg = HumanMessage(content=(
        f"Transaction Payload:\n{json.dumps(tx, indent=2)}\n\n"
        f"Layer 1 Mandatory Gate Findings ({triggered_count} triggered):\n{json.dumps(checks, indent=2)}\n\n"
        "Decide whether deeper legal/KYC investigation is needed based on Layer 1 findings."
    ))

    return {
        "layer1_checks": checks,
        "references_map": refs_map,
        "messages": [handoff_msg]
    }


def layer2_autonomous_agent_node(state: HybridAgentState) -> dict:
    """LAYER 2: Autonomous AI Detective decides whether to dig deeper with tools or finish."""
    log_debug("LAYER 2 [Autonomous Agent]: Evaluating Layer 1 findings & deciding next tool call...")
    sys_msg = SystemMessage(content=("You are Layer 2: an Autonomous Banking KYC/AML Investigation Agent.\n"
                                     "Layer 1 (Deterministic Gate) has already run mandatory sanctions, FATF, and HKD threshold checks.\n"
                                     "Your job is to decide whether deeper legal/KYC investigation is warranted:\n"
                                     "- If ALL Layer 1 checks show triggered=False AND account_status is Active: DO NOT call any tools. Immediately reply 'LAYER2_CLEAN_STOP'.\n"
                                     "- If ANY Layer 1 check has triggered=True OR account is Dormant/Corporate: you MUST call BOTH tools before finishing:\n"
                                     "  1. Call `check_enhanced_cdd_requirements` to retrieve HKMA enhanced due diligence rules.\n"
                                     "  2. Call `search_kyc_regulation` (for example with query 'beneficial owner customer due diligence monitoring') "
                                     "to retrieve supporting FinCEN statutory clauses from the regulatory corpus.\n"
                                     "Only after BOTH tools have been executed and returned results, reply 'LAYER2_INVESTIGATION_DONE'."))
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            log_debug(f"  -> LAYER 2 AGENT CALLED TOOL: {tc['name']}({tc['args']})")
    else:
        log_debug("  -> LAYER 2 AGENT DECIDED NO FURTHER TOOLS NEEDED.")
    return {"messages": [response]}


layer2_tool_node = ToolNode(LAYER2_TOOLS)


def layer3_guardrail_synthesis_node(state: HybridAgentState) -> dict:
    """LAYER 3: Synthesizes Layer 1 + Layer 2 evidence and enforces Hard Compliance Overrides."""
    log_debug("LAYER 3 [Guardrail Synthesizer]: Compiling final report & applying hard compliance floors...")
    refs_map = dict(state["references_map"])
    layer2_trace = []

    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            layer2_trace.append({"tool": msg.name, "output": msg.content})
            try:
                data = json.loads(msg.content)
                if msg.name == "search_kyc_regulation" and isinstance(data, list):
                    for item in data:
                        refs_map[item["ref_id"]] = {"title": f"Regulatory Clause {item['ref_id']}",
                            "source": item["source"], "text": item["text"]}
                elif msg.name == "check_enhanced_cdd_requirements" and "ref_id" in data:
                    refs_map[data["ref_id"]] = {"title": "HKMA Enhanced Due Diligence (EDD) Requirements",
                        "source": data["source"], "text": " ".join(data.get("requirements", []))}
            except Exception:
                pass

    triggered_layer1 = [c for c in state["layer1_checks"] if c["triggered"]]

    if not triggered_layer1 and not layer2_trace:
        log_debug("LAYER 3 FAST-PATH: Clean transaction verified across Layer 1 & Layer 2.")
        return {"report": {"overall_risk": "LOW", "violations": [],
            "recommended_action": "Clear transaction — passed all Layer 1 mandatory checks and Layer 2 agent screening."},
            "layer2_trace": layer2_trace, "references_map": refs_map, "override_applied": False}

    structured_llm = llm.with_structured_output(RiskReport)
    prompt = [SystemMessage(content=("You are Layer 3: Compliance Report Synthesizer.\n"
                                     f"Valid reference_ids: {list(refs_map.keys())}.\n"
                                     "Create a ViolationItem (VIO-1, VIO-2, ...) for every triggered Layer 1 rule and any supporting Layer 2 KYC/EDD finding.\n"
                                     "In 'reference_id', output ONLY an exact key from Valid reference_ids.")),
        HumanMessage(content=(f"Transaction: {json.dumps(state['transaction'])}\n\n"
                              f"Triggered Layer 1 Rules: {json.dumps(triggered_layer1)}\n\n"
                              f"All Verified References (Layer 1 + Layer 2): {json.dumps(refs_map)}"))]
    report = structured_llm.invoke(prompt).model_dump()

    # HARD COMPLIANCE OVERRIDE (Deterministic Guardrail)
    override_applied = False
    report["violations"] = [v for v in report["violations"] if v["reference_id"] in refs_map]

    if "UN-OFAC-SANCTIONS" in refs_map and report["overall_risk"] != "CRITICAL":
        report["overall_risk"] = "CRITICAL"
        override_applied = True
    elif triggered_layer1 and report["overall_risk"] == "LOW":
        report["overall_risk"] = "HIGH"
        override_applied = True

    return {"report": report, "layer2_trace": layer2_trace, "references_map": refs_map,
        "override_applied": override_applied}


# Compile 3-Layer Hybrid Graph
workflow = StateGraph(HybridAgentState)
workflow.add_node("layer1_gate", layer1_mandatory_gate_node)
workflow.add_node("layer2_agent", layer2_autonomous_agent_node)
workflow.add_node("layer2_tools", layer2_tool_node)
workflow.add_node("layer3_guardrail", layer3_guardrail_synthesis_node)

workflow.set_entry_point("layer1_gate")
workflow.add_edge("layer1_gate", "layer2_agent")
workflow.add_conditional_edges("layer2_agent", tools_condition,
    {"tools": "layer2_tools", "__end__": "layer3_guardrail"})
workflow.add_edge("layer2_tools", "layer2_agent")
workflow.add_edge("layer3_guardrail", END)

agent = workflow.compile()


# =====================================================================
# 5. NICEGUI WEB INTERFACE (Layer 1 -> Layer 2 -> Layer 3 Visualization)
# =====================================================================
SEVERITY_COLORS = {"LOW": "green", "MEDIUM": "amber", "HIGH": "orange", "CRITICAL": "red", "PASS": "green"}
BANK_OPTIONS = [
    # Sanctioned Banks (Trigger UN-OFAC-SANCTIONS -> CRITICAL)
    "Bank Melli",
    "Bank Sepah",
    "Bank Saderat",
    "VTB Bank",
    "Sberbank",
    "Kumgang Bank",
    "Foreign Trade Bank",
    "Central Bank of Syria",
    # Clean Commercial Banks (Pass Sanctions -> Test HIGH / LOW)
    "HSBC Hong Kong",
    "DBS Bank (Hong Kong)",
    "Bank of China (Hong Kong)",
    "Standard Chartered Bank",
    "Hang Seng Bank"
]


@ui.page("/")
def main_page():
    ui.colors(primary="#1e3a8a", secondary="#0f766e", accent="#dc2626")

    with ui.header().classes("items-center justify-between px-6 py-3 bg-slate-900 text-white shadow-md"):
        with ui.row().classes("items-center gap-3"):
            ui.icon("security", size="md").classes("text-teal-400")
            ui.label("3-Layer Guardrailed Hybrid KYC/AML Agent").classes("text-lg font-bold tracking-wide")
        ui.badge(f"OpenSanctions: {len(SANCTIONS_DB):,} | KYC Clauses: {len(KYC_DOCS)}", color="teal")

    with ui.row().classes("w-full max-w-7xl mx-auto p-6 gap-6 items-start no-wrap"):
        with ui.card().classes("w-1/3 p-5 shadow-md border border-slate-200"):
            ui.label("Wire / Payment Instruction").classes("text-base font-bold text-slate-800 mb-2")

            from_bank_input = ui.select(
                BANK_OPTIONS,
                label="From Bank (Remitting)",
                value="Bank Melli",
                with_input=True,
                new_value_mode="add-unique"
            ).classes("w-full")

            from_acct_input = ui.select(
                [
                    "External Corporate Account (Iran)",
                    "External Corporate Account (Russia)",
                    "External Retail Account (Singapore)",
                    "HK Active Account (3+ years)",
                    "HK Dormant Account (210 days inactive)"
                ],
                label="From Account Profile",
                value="External Corporate Account (Iran)",
                with_input=True,
                new_value_mode="add-unique"
            ).classes("w-full")

            from_acct_no_input = ui.input(
                "From Account No. / IBAN",
                value="IR060170000000100324200001"
            ).classes("w-full")

            to_bank_input = ui.select(
                BANK_OPTIONS,
                label="To Bank (Beneficiary)",
                value="HSBC Hong Kong",
                with_input=True,
                new_value_mode="add-unique"
            ).classes("w-full")

            to_acct_input = ui.select(
                [
                    "HK Dormant Account (210 days inactive)",
                    "HK Active Account (3+ years)",
                    "HK Newly Opened Corporate Account",
                    "External Corporate Account (Iran)",
                    "External Retail Account (UK)"
                ],
                label="To Account Profile",
                value="HK Dormant Account (210 days inactive)",
                with_input=True,
                new_value_mode="add-unique"
            ).classes("w-full")

            to_acct_no_input = ui.input(
                "To Account No. (HK)",
                value="004-512-889012-838"
            ).classes("w-full")

            type_input = ui.select(
                ["Cross-Border SWIFT Wire", "Local CHATS / RTGS", "Local FPS Transfer"],
                label="Transfer Type",
                value="Cross-Border SWIFT Wire"
            ).classes("w-full")

            amount_input = ui.number("Amount (HKD)", value=185000, min=1, format="%.0f").classes("w-full")

            run_btn = ui.button("Run 3-Layer Hybrid Agent", icon="play_arrow").classes("w-full mt-4 py-2 font-semibold")
            spinner = ui.spinner("dots", size="lg", color="primary").classes("mx-auto mt-2 hidden")

        results_container = ui.column().classes("w-2/3 gap-4")
        with results_container:
            with ui.card().classes("w-full p-6 bg-slate-50 border border-dashed border-slate-300"):
                ui.label("Click 'Run 3-Layer Hybrid Agent' to view Layer 1 Mandatory Statutory Checks, Layer 2 Autonomous Detective Reasoning, and Layer 3 Guardrailed Bookmarks.").classes("text-slate-500")

        async def execute_triage():
            log_debug("[UI CLICK] 'Run 3-Layer Hybrid Agent' button clicked!")
            try:
                from_bank = (from_bank_input.value or "").strip()
                to_bank = (to_bank_input.value or "").strip()
                log_debug(
                    f"[UI CLICK] Inputs -> From='{from_bank}' ({from_acct_input.value}) "
                    f"-> To='{to_bank}' ({to_acct_input.value}), Type='{type_input.value}', Amount={amount_input.value}"
                )

                if not from_bank or not to_bank:
                    log_debug("[UI CLICK] Aborted: From Bank or To Bank is empty.")
                    ui.notify("Please select or enter both From Bank and To Bank", type="warning")
                    return

                run_btn.disable()
                spinner.classes(remove="hidden")
                results_container.clear()
                with results_container:
                    ui.label("Executing Layer 1 Mandatory Gate -> Layer 2 Autonomous Agent -> Layer 3 Guardrail...").classes("text-slate-600 italic")

                tx_payload = {
                    "from_bank": from_bank,
                    "from_account": (from_acct_input.value or "").strip(),
                    "from_account_no": (from_acct_no_input.value or "").strip(),
                    "to_bank": to_bank,
                    "to_account": (to_acct_input.value or "").strip(),
                    "to_account_no": (to_acct_no_input.value or "").strip(),
                    "type": type_input.value,
                    "amount_hkd": int(amount_input.value or 0)
                }

                log_debug("[UI CLICK] Invoking LangGraph workflow in background thread...")
                final_state = await run.io_bound(agent.invoke, {"transaction": tx_payload, "messages": []})
                log_debug("[UI CLICK] LangGraph workflow completed successfully!")

            except Exception as err:
                import traceback
                tb = traceback.format_exc()
                log_debug(f"[UI CLICK ERROR] {repr(err)}\n{tb}")
                spinner.classes(add="hidden")
                run_btn.enable()
                results_container.clear()
                with results_container:
                    with ui.card().classes("w-full p-5 border-l-4 border-red-600 bg-red-50"):
                        ui.label("Execution Error").classes("font-bold text-red-800")
                        ui.label(str(err)).classes("text-sm text-red-700 font-mono mt-1")
                return

            spinner.classes(add="hidden")
            run_btn.enable()
            results_container.clear()
            log_debug("[UI CLICK] Rendering Layer 1, Layer 2, and Layer 3 UI cards...")

            layer1_checks = final_state["layer1_checks"]
            layer2_trace = final_state.get("layer2_trace", [])
            report = final_state["report"]
            refs_map = final_state["references_map"]
            violations = report.get("violations", [])
            risk = report.get("overall_risk", "LOW")

            with results_container:
                triggered_l1 = sum(1 for c in layer1_checks if c["triggered"])
                with ui.card().classes("w-full p-5 shadow-sm border-l-4 border-indigo-700"):
                    with ui.row().classes("w-full items-center justify-between mb-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("gavel", color="indigo")
                            ui.label("LAYER 1: Mandatory Statutory Compliance Gate (Deterministic Code)").classes("font-bold text-slate-800")
                        ui.badge(f"{triggered_l1} / {len(layer1_checks)} Triggered", color="red" if triggered_l1 else "green")

                    for chk in layer1_checks:
                        is_trig = chk["triggered"]
                        row_bg = "bg-red-50 border-red-200" if is_trig else "bg-slate-50 border-slate-200"
                        with ui.row().classes(f"w-full items-center justify-between p-2 rounded border {row_bg} mb-1"):
                            with ui.column().classes("gap-0"):
                                ui.label(f"{chk['rule']} [{chk['ref_id']}]").classes("text-sm font-semibold text-slate-800")
                                ui.label(chk["detail"]).classes("text-xs text-slate-600")
                            ui.badge("TRIGGERED" if is_trig else "PASS", color="red" if is_trig else "green")

                with ui.card().classes("w-full p-5 shadow-sm border-l-4 border-teal-600 bg-slate-900 text-white"):
                    with ui.row().classes("w-full items-center justify-between mb-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.icon("psychology", color="teal")
                            ui.label("LAYER 2: Autonomous AI Detective Investigation (ReAct Tool Loop)").classes("font-bold text-teal-300")
                        ui.badge(f"{len(layer2_trace)} Autonomous Tool Calls", color="teal")

                    if not layer2_trace:
                        ui.label("Decision: Layer 1 reported 0 triggers on an active account. Agent autonomously skipped deep legal RAG to save latency & cost.").classes("text-xs font-mono text-slate-300")
                    else:
                        ui.label("Decision: Red flags detected in Layer 1. Agent autonomously invoked legal & EDD tools to dig deeper:").classes("text-xs text-slate-300 mb-2")
                        for idx, step in enumerate(layer2_trace, 1):
                            ui.label(f"Step {idx} -> Agent called `{step['tool']}`").classes("text-xs font-mono text-teal-300 font-bold")
                            ui.label(step["output"][:260] + ("..." if len(step["output"]) not in range(0, 261) else "")).classes("text-xs font-mono text-slate-300 mb-2 pl-3 border-l-2 border-teal-500")

                with ui.card().classes("w-full p-5 shadow-sm border-l-4 border-blue-800"):
                    with ui.row().classes("w-full items-center justify-between"):
                        with ui.row().classes("items-center gap-3"):
                            ui.label("LAYER 3 Final Risk Rating:").classes("font-bold text-slate-800")
                            ui.badge(risk, color=SEVERITY_COLORS.get(risk, "grey")).classes("text-sm px-3 py-1 font-bold")
                        ui.badge(
                            "Hard Override Applied" if final_state["override_applied"] else "Guardrail Verified",
                            color="orange" if final_state["override_applied"] else "green"
                        )
                    ui.label(f"Recommended Action: {report['recommended_action']}").classes("mt-2 font-medium text-slate-800")

                with ui.card().classes("w-full p-4 bg-slate-50 border border-slate-200"):
                    ui.label("Violation Bookmarks (Click to Jump)").classes("text-xs font-bold uppercase tracking-wider text-slate-500 mb-2")
                    if not violations:
                        ui.label("No regulatory violations detected.").classes("text-green-700 font-medium")
                    else:
                        with ui.row().classes("gap-2 flex-wrap"):
                            for v in violations:
                                vid = v["violation_id"]
                                sev_color = SEVERITY_COLORS.get(v["severity"], "orange")
                                ui.button(
                                    f"#{vid}: {v['title']} -> [{v['reference_id']}]",
                                    icon="bookmark",
                                    on_click=lambda target=vid: ui.run_javascript(
                                        f"document.getElementById('{target}').scrollIntoView({{behavior: 'smooth', block: 'center'}});"
                                    )
                                ).props(f"outline color={sev_color} size=sm")

                if violations:
                    ui.label("Detected Violations").classes("text-base font-bold text-slate-800 mt-1")
                    for v in violations:
                        vid = v["violation_id"]
                        ref_id = v["reference_id"]
                        sev_color = SEVERITY_COLORS.get(v["severity"], "orange")
                        with ui.card().props(f'id="{vid}"').classes("w-full p-4 shadow-sm border border-slate-200"):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.badge(vid, color=sev_color).classes("font-bold")
                                    ui.label(v["title"]).classes("font-bold text-slate-800")
                                ui.button(
                                    f"Jump to Reference: {ref_id}", icon="arrow_downward",
                                    on_click=lambda r_target=f"ref-{ref_id}": ui.run_javascript(
                                        f"document.getElementById('{r_target}').scrollIntoView({{behavior: 'smooth', block: 'center'}});"
                                    )
                                ).props("flat dense color=primary size=sm")
                            ui.label(v["explanation"]).classes("text-sm text-slate-700 mt-2")

                if refs_map:
                    ui.label("Regulatory & Watchlist References (Layer 1 + Layer 2)").classes(
                        "text-base font-bold text-slate-800 mt-2"
                    )
                    cited_ids = {v["reference_id"] for v in violations}
                    for ref_id, info in refs_map.items():
                        is_cited = ref_id in cited_ids
                        border_style = "border-l-4 border-red-600 bg-red-50/30" if is_cited else "border border-slate-200"
                        with ui.card().props(f'id="ref-{ref_id}"').classes(f"w-full p-4 shadow-sm {border_style}"):
                            with ui.row().classes("w-full items-center justify-between"):
                                with ui.row().classes("items-center gap-2"):
                                    ui.badge(ref_id, color="red" if is_cited else "slate").classes("font-mono")
                                    ui.label(info["title"]).classes("font-bold text-slate-800")
                                if is_cited:
                                    ui.badge("CITED IN VIOLATION", color="red").props("outline")
                            ui.label(f"Source: {info['source']}").classes("text-xs text-slate-500 mt-1")
                            ui.label(info["text"]).classes(
                                "text-sm text-slate-700 mt-2 bg-white p-3 rounded border border-slate-100")
            log_debug("[UI CLICK] UI rendering complete!")


        run_btn.on_click(execute_triage)   # bound at page-build time, after the handler is defined
        log_debug("[UI INIT] main_page() rendered and run_btn.on_click bound.")



def free_port_windows(port: int) -> None:
    """Terminate any process listening on ``port`` (Windows only, best effort)."""
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"], text=True, errors="ignore"
        )
    except Exception:
        return
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local_address, state, pid_text = parts[1], parts[3], parts[4]
        if state.upper() != "LISTENING" or not local_address.endswith(f":{port}"):
            continue
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid == os.getpid():
            continue
        log_debug(f"[PORT] Releasing port {port} from stale process {pid}...")
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        )

if __name__ in {"__main__", "__mp_main__"}:
    is_hf_space = os.getenv("SPACE_ID") is not None
    PORT = 7860 if is_hf_space else 8085
    if sys.platform == "win32" and not is_hf_space:
        free_port_windows(PORT)
    ui.run(title="3-Layer Guardrailed Hybrid KYC/AML Agent", host="0.0.0.0", port=PORT,
        show=not is_hf_space, reload=False, reconnect_timeout=30.0)
