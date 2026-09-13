"""GCP adapter. Implement upload/download/push_image for Lab 1.

SDK:  pip install google-cloud-storage google-cloud-aiplatform
Docs: storage.Client for GCS; Artifact Registry push goes through `docker push` after
      `gcloud auth configure-docker <region>-docker.pkg.dev`.

Hints for Lab 1:
  * BLOB_URI looks like gs://bucket/prefix — parse it here, never in src/.
  * Artifact Registry paths are region-scoped:
        <region>-docker.pkg.dev/<project>/<repo>/<image>
    A common first failure is pushing to gcr.io out of habit; it is a different service.
  * push_image must return the digest reference, not the tag.
  * GCP calls them labels, not tags, and they must be lowercase with no spaces.
    cfg.tags(1) already satisfies that constraint — do not "improve" the values.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from cloudlayer.base import CloudAdapter

_DIGEST_RE = re.compile(r"digest:\s*(sha256:[0-9a-f]{64})")


def _split_gs_uri(uri: str) -> tuple[str, str]:
    without_scheme = uri.removeprefix("gs://")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


class GcpAdapter(CloudAdapter):
    def upload(self, local_path: str, key: str) -> str:
        from google.cloud import storage

        bucket_name, prefix = _split_gs_uri(self.cfg.blob_uri)
        full_key = f"{prefix.rstrip('/')}/{key}" if prefix else key
        client = storage.Client(project=self.cfg.project_id)
        blob = client.bucket(bucket_name).blob(full_key)
        blob.upload_from_filename(local_path)
        return f"gs://{bucket_name}/{full_key}"

    def download(self, uri: str, local_path: str) -> None:
        from google.cloud import storage

        bucket_name, key = _split_gs_uri(uri)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        client = storage.Client(project=self.cfg.project_id)
        client.bucket(bucket_name).blob(key).download_to_filename(local_path)

    def push_image(self, local_tag: str) -> str:
        registry = self.cfg.container_registry.rstrip("/")
        registry_host = registry.split("/", 1)[0]
        repo_path, _, tag = local_tag.rpartition(":")
        remote_tag = f"{registry}/{repo_path}:{tag or 'latest'}"

        subprocess.run(
            ["gcloud", "auth", "configure-docker", registry_host, "--quiet"],
            check=True,
        )
        subprocess.run(["docker", "tag", local_tag, remote_tag], check=True)
        # Which stream carries "digest: sha256:..." depends on the docker CLI's progress
        # writer (buildkit vs. classic, TTY vs. not) and isn't consistent across versions —
        # merge stderr into stdout so the search doesn't depend on that detail.
        result = subprocess.run(
            ["docker", "push", remote_tag],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

        match = _DIGEST_RE.search(result.stdout)
        if not match:
            raise RuntimeError(f"could not parse digest from push output:\n{result.stdout}")
        return f"{registry}/{repo_path}@{match.group(1)}"

    # submit_training / register_model  -> Lab 2 (Vertex custom training + Model Registry)
    # deploy / invoke                   -> Lab 3 (Vertex Endpoint)
    # emit_metric                       -> Lab 4 (Cloud Monitoring time series)
    # generate                          -> Lab 5 (managed LLM endpoint; read usageMetadata for tokens)
    # teardown                          -> Lab 5 (filter resources by label)
