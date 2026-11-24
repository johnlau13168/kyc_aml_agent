"""Entry point for the 3-Layer Guardrailed Hybrid KYC/AML Agent.

The implementation lives in ``kyc_aml_compliance.agent.py`` (the filename is
kept as published).  This launcher exists so the application can be started
with the conventional ``python app.py`` command and so that platforms which
expect an ``app.py`` entry point (Docker, Hugging Face Spaces) work unchanged.

It also performs a dependency preflight so that running the application with an
interpreter that lacks the requirements produces an actionable message instead
of a bare ``ModuleNotFoundError`` traceback.
"""

import importlib.util
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "kyc_aml_compliance.agent.py")

# import name -> distribution name, in the order they are imported by the agent
REQUIRED = {
    "nicegui": "nicegui",
    "pydantic": "pydantic",
    "langchain_core": "langchain-core",
    "langchain_community": "langchain-community",
    "langgraph": "langgraph",
    "huggingface_hub": "huggingface-hub",
    "llama_cpp": "llama-cpp-python",
}


def missing_requirements():
    missing = []
    for module, distribution in REQUIRED.items():
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append((module, distribution))
    return missing


def report_missing(missing):
    print("Cannot start the KYC/AML agent: required packages are not installed.\n")
    print(f"Interpreter: {sys.executable}")
    print(f"Version:     {sys.version.split()[0]}\n")
    print("Missing packages:")
    for module, distribution in missing:
        print(f"  - {distribution}  (import name: {module})")
    print("\nInstall them into this interpreter with:\n")
    print(f'  "{sys.executable}" -m pip install -r requirements.txt\n')
    print("If you use a virtual environment, activate it first or point your")
    print("editor at .venv\\Scripts\\python.exe (Windows) or .venv/bin/python.")
    print("\nNote: llama-cpp-python needs Python 3.10-3.12. On newer versions it")
    print("must be compiled from source.")


if __name__ == "__main__":
    missing = missing_requirements()
    if missing:
        report_missing(missing)
        raise SystemExit(1)
    runpy.run_path(TARGET, run_name="__main__")
