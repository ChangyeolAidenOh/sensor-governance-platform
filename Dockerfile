# Public demo image (demo-public branch): Streamlit app + the skills_detect server subprocess, stored-result retrieval.
# Build context = the demo-public worktree; model and reference files come from cloudrun/assets (cloudrun/prepare_image_assets.sh).
FROM python:3.11-slim

ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 FASTMCP_CHECK_FOR_UPDATES=off \
    SGP_PUBLIC_DEMO=1 SGP_DEMO_MANIFEST=outputs/demo_presets_public.json SGP_DEMO_SYNTHETIC_PRESETS=11 \
    SGP_DETECTOR_DIR=models/detector_v1 SGP_DIAGNOSER_DIR=models/diagnoser_v1 \
    SGP_GATE_CALIB=models/gate_v2/gate_calibration.json SGP_GATE_REF=data/processed/gate_reference_v1.parquet \
    PORT=8080

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements_public.txt ./
RUN pip install -r requirements_public.txt

COPY core_pipeline ./core_pipeline
COPY demo ./demo
COPY .streamlit ./.streamlit
COPY outputs/demo_presets.json outputs/demo_presets_public.json outputs/public_demo_questions.json outputs/fault_to_guidance_map.json ./outputs/
COPY outputs/model_meta ./outputs/model_meta
COPY models/gate_v2 ./models/gate_v2
COPY models/retrieval_v1/tau.json models/retrieval_v1/index_meta.json ./models/retrieval_v1/
COPY cloudrun/assets/models/detector_v1 ./models/detector_v1
COPY cloudrun/assets/models/diagnoser_v1 ./models/diagnoser_v1
COPY cloudrun/assets/data/processed/gate_reference_v1.parquet ./data/processed/gate_reference_v1.parquet

RUN cd /tmp && python -c "import core_pipeline.skills.servers.skills_detect, demo.runtime, demo.public_guard, core_pipeline.skills.public_skills; print('imports ok')"

EXPOSE 8080
CMD exec streamlit run demo/app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false
