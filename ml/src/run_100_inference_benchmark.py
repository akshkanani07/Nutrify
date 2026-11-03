"""Nutrify AI — Production 100-Random-Inference Stress & Performance Benchmark."""

import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ml.src.production_pipeline import ProductionInferenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nutrify.benchmark100")


def get_memory_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    # Windows memory query via ctypes
    try:
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 350.0


def run_benchmark():
    service = ProductionInferenceService(
        checkpoint_path="ml/models/production/nutrition_model_650_best.pt",
        candidates_csv="ml/data/raw/candidates.csv",
        device="cpu",
        confidence_threshold=0.50,
        num_threads=4,
    )

    test_csv = Path("ml/data/annotations/test.csv")
    images_base = Path("ml/data/images")

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        all_rows = [row for row in reader]

    logger.info(f"Loaded {len(all_rows)} test split records.")

    # Select 100 random distinct classes
    random.seed(42)
    classes_seen = set()
    selected_samples = []

    # Shuffle to ensure randomness
    shuffled_rows = list(all_rows)
    random.shuffle(shuffled_rows)

    for row in shuffled_rows:
        label = row["label"]
        img_path = images_base / row["image_path"]
        if label not in classes_seen and img_path.exists():
            classes_seen.add(label)
            selected_samples.append({
                "label": label,
                "image_path": str(img_path),
            })
            if len(selected_samples) >= 100:
                break

    logger.info(f"Selected {len(selected_samples)} diverse evaluation samples.")

    # Warmup
    warmup_img = selected_samples[0]["image_path"]
    for _ in range(3):
        _ = service.predict(warmup_img)

    mem_before_mb = get_memory_mb()

    latencies = []
    top1_correct = 0
    top5_correct = 0
    tier_counts = {"high": 0, "medium": 0, "low": 0}
    results = []

    logger.info("Executing 100 inference evaluations...")

    for i, sample in enumerate(selected_samples, 1):
        t0 = time.perf_counter()
        pred = service.predict(sample["image_path"], top_k=5)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        true_label = sample["label"]
        pred_label = pred["food_name"]
        candidate_labels = [c["food_name"] for c in pred["candidates"]]

        is_top1 = (pred_label.lower() == true_label.lower())
        is_top5 = any(c.lower() == true_label.lower() for c in candidate_labels)

        if is_top1:
            top1_correct += 1
        if is_top5:
            top5_correct += 1

        tier = pred.get("confidence_tier", "low")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

        # Verify nutrition sanity
        assert pred["calories"] >= 0
        assert pred["protein"] >= 0
        assert pred["carbs"] >= 0
        assert pred["fat"] >= 0
        assert pred["fiber"] >= 0

        results.append({
            "sample_id": i,
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": pred["confidence"],
            "confidence_tier": tier,
            "latency_ms": round(elapsed_ms, 2),
            "is_top1_correct": is_top1,
            "is_top5_correct": is_top5,
            "calories": pred["calories"],
            "protein": pred["protein"],
            "carbs": pred["carbs"],
            "fat": pred["fat"],
            "fiber": pred["fiber"],
        })

    mem_after_mb = get_memory_mb()

    mean_lat = sum(latencies) / len(latencies)
    sorted_lat = sorted(latencies)
    p50_lat = sorted_lat[len(sorted_lat) // 2]
    p95_lat = sorted_lat[int(len(sorted_lat) * 0.95)]
    min_lat = sorted_lat[0]
    max_lat = sorted_lat[-1]

    top1_acc = (top1_correct / len(selected_samples)) * 100.0
    top5_acc = (top5_correct / len(selected_samples)) * 100.0

    report = {
        "benchmark_timestamp": time.time(),
        "total_samples": len(selected_samples),
        "total_classes_tested": len(classes_seen),
        "top1_accuracy_percent": round(top1_acc, 2),
        "top5_accuracy_percent": round(top5_acc, 2),
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "min": round(min_lat, 2),
            "max": round(max_lat, 2),
        },
        "throughput_fps": round(1000.0 / mean_lat, 2),
        "memory_rss_mb": {
            "before": round(mem_before_mb, 2),
            "after": round(mem_after_mb, 2),
            "delta": round(mem_after_mb - mem_before_mb, 2),
        },
        "confidence_tier_distribution": tier_counts,
        "sample_evaluations": results,
    }

    out_file = Path("ml/data/reports/production_hardening_100_inferences.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"--- 100 Random Inference Results ---")
    logger.info(f"Top-1 Accuracy: {top1_acc:.2f}%")
    logger.info(f"Top-5 Accuracy: {top5_acc:.2f}%")
    logger.info(f"Latency: Mean={mean_lat:.2f}ms, P50={p50_lat:.2f}ms, P95={p95_lat:.2f}ms")
    logger.info(f"Throughput: {report['throughput_fps']} FPS")
    logger.info(f"Memory RSS: {mem_after_mb:.1f} MB (Delta: {mem_after_mb - mem_before_mb:.2f} MB)")
    logger.info(f"Report written to: {out_file}")

    return report


if __name__ == "__main__":
    run_benchmark()
