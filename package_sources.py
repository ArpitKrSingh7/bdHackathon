"""Build a shareable source/artifact bundle without duplicating the raw archive."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "internet_ads_source_bundle.zip"
MANIFEST_PATH = PROJECT_DIR / "artifacts" / "report" / "source_bundle_manifest.json"
SOURCE_FILES = [
    "AGENTS.md",
    "internet_ads_hackathon.ipynb",
    "internet_ads_hackathon_report.pdf",
    "requirements.txt",
    "progress.md",
    "team_handoff.md",
    "hackathon_utils.py",
    "preprocess_dataset.py",
    "association_mining.py",
    "classification_clustering.py",
    "pyspark_benchmark.py",
    "report_figures.py",
    "experiment_tracking.py",
    "log_experiments.py",
    "artifacts/mlflow.db",
    "build_notebook.py",
    "build_report.py",
    "package_sources.py",
    "artifacts/report/figure_manifest.json",
]


def _write_readme(zip_file: ZipFile) -> None:
    text = """Internet Ads Big Data Hackathon source bundle

Contents: source scripts, executed notebook, final PDF report, dependency versions, progress/handoff notes, generated analysis outputs, and the local MLflow SQLite database/artifacts. The original raw data is not bundled; download it from:
https://archive.ics.uci.edu/static/public/51/internet+advertisements.zip

Extract the archive. Put the UCI source files in data/raw/ (ad.data, ad.names, ad.DOCUMENTATION, internet-advertisements.zip). Install the packages in requirements.txt, then run from the bdHackathon directory:

  python3 preprocess_dataset.py
  python3 association_mining.py
  python3 classification_clustering.py
  python3 pyspark_benchmark.py
  python3 report_figures.py
  python3 log_experiments.py
  python3 build_notebook.py
  python3 build_report.py
  python3 package_sources.py

The captured benchmark completed both PySpark workloads in local[2]. Spark was slower than the matched local tasks on this 3,279-row dataset; startup and data-conversion/collection costs are recorded separately. If a later environment blocks the Java gateway socket, the benchmark records that status without inventing task timings.

Internet Ads has no timestamps, sessions, or natural event order. The synthetic GSP sequence demonstration is not a dataset finding and does not fulfill dataset-level sequential mining; ordered event data or an instructor-approved interpretation would be needed for that requirement.
"""
    zip_file.writestr("README_source_bundle.txt", text)


def build_bundle() -> dict:
    entries = []
    with ZipFile(OUTPUT_PATH, "w", compression=ZIP_DEFLATED, compresslevel=7) as archive:
        _write_readme(archive)
        for relative in SOURCE_FILES:
            path = PROJECT_DIR / relative
            if path.is_file():
                archive.write(path, arcname=relative)
                entries.append(relative)
        for directory in ("artifacts/processed", "artifacts/association", "artifacts/models", "artifacts/spark", "artifacts/report/figures", "artifacts/mlflow"):
            for path in sorted((PROJECT_DIR / directory).rglob("*")):
                if not path.is_file() or path.suffix == ".pyc":
                    continue
                relative = path.relative_to(PROJECT_DIR).as_posix()
                archive.write(path, arcname=relative)
                entries.append(relative)
    digest = hashlib.sha256(OUTPUT_PATH.read_bytes()).hexdigest()
    manifest = {
        "bundle": OUTPUT_PATH.name,
        "bytes": OUTPUT_PATH.stat().st_size,
        "sha256": digest,
        "contains_raw_data": False,
        "files": entries,
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"bundle": str(OUTPUT_PATH), "bytes": manifest["bytes"], "sha256": digest, "file_count": len(entries) + 1}


if __name__ == "__main__":
    print(json.dumps(build_bundle(), indent=2))
