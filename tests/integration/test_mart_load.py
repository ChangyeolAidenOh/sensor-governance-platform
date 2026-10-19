"""Stage 1 S1-02: mart load criteria as a pytest (needs a live Postgres; set SGP_PG_DSN)."""
import os
import subprocess
import sys

import pytest

DSN = os.environ.get("SGP_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="SGP_PG_DSN not set")


def test_load_is_idempotent_and_passes_criteria():
    for _ in range(2):   # twice: idempotency is part of the criteria (7c)
        subprocess.run([sys.executable, "scripts/load_mart.py", "--dsn", DSN], check=True)
    subprocess.run([sys.executable, "scripts/verify_mart.py", "--dsn", DSN] + (["--agent-dsn", os.environ["SGP_AGENT_DSN"]] if os.environ.get("SGP_AGENT_DSN") else []), check=True)
