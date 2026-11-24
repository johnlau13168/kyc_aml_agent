"""Entry point for the 3-Layer Guardrailed Hybrid KYC/AML Agent.

The implementation lives in ``kyc_aml_compliance.agent.py`` (the filename is
kept as published).  This launcher exists so the application can be started
with the conventional ``python app.py`` command and so that platforms which
expect an ``app.py`` entry point (Docker, Hugging Face Spaces) work unchanged.
"""

import os
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "kyc_aml_compliance.agent.py")

if __name__ == "__main__":
    runpy.run_path(TARGET, run_name="__main__")
