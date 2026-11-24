"""
Script to scan and extract all metrics from all reports across the repository.
"""
import os
import re
import json
from pathlib import Path

PROJECT_ROOT = Path("C:/Nutrify")
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"
DATA_REPORTS_DIR = PROJECT_ROOT / "ml" / "data" / "reports"
ARTIFACTS_DIR = Path(r"C:\Users\AkshKanani\.gemini\antigravity\brain\b7d6aba7-8151-4941-a38f-feefc1012f2a")

report_files = []

# Collect all markdown and json reports
for p in REPORTS_DIR.glob("*.*"):
    if p.suffix in (".md", ".json"):
        report_files.append(("ml/reports", p))

for p in DATA_REPORTS_DIR.glob("*.*"):
    if p.suffix in (".md", ".json"):
        report_files.append(("ml/data/reports", p))

for p in ARTIFACTS_DIR.glob("*.md"):
    report_files.append(("artifacts", p))

print(f"Total reports found: {len(report_files)}")

results = {}

for cat, file_path in sorted(report_files, key=lambda x: str(x[1])):
    fname = file_path.name
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue
    
    entry = {"category": cat, "path": str(file_path), "size_bytes": file_path.stat().st_size}
    metrics = {}
    
    # If JSON, try to extract structured metrics
    if file_path.suffix == ".json":
        try:
            data = json.loads(content)
            entry["json_keys"] = list(data.keys()) if isinstance(data, dict) else "list"
            # Look for common metrics
            if isinstance(data, dict):
                for k in ["top1_accuracy", "top5_accuracy", "measured_top1_accuracy", "measured_top5_accuracy",
                          "precision", "macro_precision", "recall", "macro_recall", "f1", "macro_f1",
                          "ece", "ece_score", "ece_calibration_error", "mean_ms", "latency_ms",
                          "total_samples", "test_samples", "num_classes", "train_loss", "val_loss",
                          "checkpoint", "checkpoint_path"]:
                    if k in data:
                        metrics[k] = data[k]
                if "overall_metrics" in data:
                    metrics["overall_metrics"] = data["overall_metrics"]
                if "empirical_metrics" in data:
                    metrics["empirical_metrics"] = data["empirical_metrics"]
                if "inference_latency_cpu" in data:
                    metrics["inference_latency_cpu"] = data["inference_latency_cpu"]
        except Exception:
            pass
    
    # Text search regex patterns
    patterns = {
        "top1": r"(?:top[- ]?1(?:\s+accuracy)?[:\s|]+)([\d\.]+%?)",
        "top5": r"(?:top[- ]?5(?:\s+accuracy)?[:\s|]+)([\d\.]+%?)",
        "precision": r"(?:precision[:\s|]+)([\d\.]+%?)",
        "recall": r"(?:recall[:\s|]+)([\d\.]+%?)",
        "f1": r"(?:f1[- ]?(?:score)?[:\s|]+)([\d\.]+%?)",
        "ece": r"(?:ece[:\s|]+)([\d\.]+)",
        "latency": r"(?:latency[:\s|]+)([\d\.]+\s*ms)",
        "dataset_size": r"(?:total (?:images|samples)[:\s|]+)([\d\,]+)",
        "train_size": r"(?:train(?:ing)? (?:images|samples)[:\s|]+)([\d\,]+)",
        "val_size": r"(?:val(?:idation)? (?:images|samples)[:\s|]+)([\d\,]+)",
        "checkpoint": r"(nutrition_model_[^\s`\)]+\.pt)",
    }
    
    text_matches = {}
    for mname, pat in patterns.items():
        matches = re.findall(pat, content, re.IGNORECASE)
        if matches:
            text_matches[mname] = matches[:5]
    
    entry["text_matches"] = text_matches
    entry["extracted_metrics"] = metrics
    results[fname] = entry

print(f"Scanned {len(results)} files.")
with open(PROJECT_ROOT / "scratch" / "scanned_metrics.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved scratch/scanned_metrics.json")
