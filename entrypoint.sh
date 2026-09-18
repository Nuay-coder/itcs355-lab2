#!/bin/sh
# Pulls raw data from the DVC remote (.dvc/config, pointing at ${BLOB_URI}/dvc) before
# training, so a remote job with no local /app/data still has something to read.
#
# Skipped when the data already exists — the `make reproduce` case, where data/ is
# bind-mounted read-only from the host. dvc would have nothing to fetch there anyway,
# and the image ships no cloud credentials for `make reproduce` to reach the remote
# with. On Vertex AI, `dvc pull` picks up the job's service account automatically via
# Application Default Credentials — no key ever enters this image.
set -eu

RAW_DATA=data/raw/sensors.csv
METRICS_OUT=reports/metrics.json

if [ ! -f "$RAW_DATA" ]; then
    echo "entrypoint: $RAW_DATA not found, pulling from DVC remote" >&2
    # No .git ships in this image (see src/train.py), and dvc requires either an SCM
    # or this local (uncommitted, machine-specific) override to run without one.
    dvc config --local core.no_scm true
    dvc pull data/raw.dvc
fi

python -m src.train "$@"

# A managed run is ephemeral — nothing written locally survives job teardown. When
# TRIAL_KEY is set (by cloudlayer/gcp.py's submit_training, for a sweep), push this
# trial's metrics to BLOB_URI so the sweep can read the result back through the same
# adapter it used to launch the job, rather than scraping container logs.
if [ -n "${TRIAL_KEY:-}" ] && [ -f "$METRICS_OUT" ]; then
    python -c "
from src import config
from cloudlayer.factory import get_adapter
import os
cfg = config.load(strict=False)
key = f\"training-jobs/{os.environ['TRIAL_KEY']}/metrics.json\"
print('entrypoint: uploaded', get_adapter(cfg).upload('$METRICS_OUT', key))
"
fi
