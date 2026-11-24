"""
Full empirical benchmark and serialization for the optimized production model.
"""
import os, sys, json, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(r"C:\Nutrify")
PROD_DIR = PROJECT_ROOT / "ml" / "models" / "production"
REPORTS_DIR = PROJECT_ROOT / "ml" / "reports"
CATALOG_PATH = PROJECT_ROOT / "ml" / "knowledge_base" / "expanded_catalog_650.json"
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.scalable_trainer import ScalableClassifierHead, ScalableFoodModel, build_backbone

device = torch.device("cpu")
torch.set_num_threads(4)

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    catalog = json.load(f)

all_classes = sorted(list(catalog["classes"].keys()))
num_classes = len(all_classes)
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for i, c in enumerate(all_classes)}

# Load candidate checkpoint
candidate_path = PROD_DIR / "candidate_model_650_realworld.pt"
cand_ckpt = torch.load(candidate_path, map_location=device, weights_only=False)

extractor, in_features, _ = build_backbone("efficientnet_b0", pretrained=False)
head = ScalableClassifierHead(in_features=in_features, num_classes=num_classes)
model = ScalableFoodModel(extractor, head, "efficientnet_b0")
model.load_state_dict(cand_ckpt["model_state_dict"])
model.eval()

# Load old baseline model (50 classes)
old_ckpt = torch.load(PROD_DIR / "nutrition_model_best.pt", map_location=device, weights_only=False)
extractor_old, in_f_old, _ = build_backbone("efficientnet_b0", pretrained=False)
head_old = ScalableClassifierHead(in_features=in_f_old, num_classes=50)
model_old = ScalableFoodModel(extractor_old, head_old, "efficientnet_b0")
if "model_state_dict" in old_ckpt:
    try:
        model_old.load_state_dict(old_ckpt["model_state_dict"])
    except Exception:
        pass
model_old.eval()

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

CLASS_MAPPING = {
    'dal_makhani': 'dal makhani',
    'shahi_paneer': 'paneer butter masala',
    'vegetable_biryani': 'biryani chicken',
    'pulao': 'fried rice',
    'jeera_rice': 'fried rice',
    'khichdi': 'dal tadka',
    'roti': 'roti chapati',
    'naan': 'naan plain',
    'kulcha': 'naan plain',
    'paratha': 'paratha aloo'
}

test_cases = []
for raw_cls, target_cls in CLASS_MAPPING.items():
    raw_folder = RAW_DIR / raw_cls
    imgs = sorted([p for p in raw_folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}])
    held_out = imgs[24:]
    for p in held_out:
        test_cases.append((p, target_cls, raw_cls))

val_100_dir = PROJECT_ROOT / "ml" / "data" / "reports" / "validation_100_images"
if val_100_dir.exists():
    for p in val_100_dir.glob("*.jpg"):
        lbl_part = p.stem.split("_", 2)[-1].replace("_", " ") if "_" in p.stem else ""
        if lbl_part in class_to_idx:
            test_cases.append((p, lbl_part, "baseline_validation"))

print(f"Total test images: {len(test_cases)}")

correct_top1 = 0
correct_top5 = 0
confidences = []
latencies = []
cluster_stats = {}
y_true = []
y_pred = []
confusions = []
failed_cases = []

Image.MAX_IMAGE_PIXELS = 25_000_000

for p, true_cls, cluster in test_cases:
    if cluster not in cluster_stats:
        cluster_stats[cluster] = {"total": 0, "top1": 0, "top5": 0, "conf_sum": 0.0}
    cluster_stats[cluster]["total"] += 1
    
    try:
        with Image.open(p) as img:
            t = val_transform(img.convert("RGB")).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Skip {p}: {e}")
        continue
    
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(t)
        probs = F.softmax(logits, dim=-1)
    lat = (time.perf_counter() - t0) * 1000.0
    latencies.append(lat)
    
    top5_indices = torch.topk(probs, 5, dim=-1).indices[0].tolist()
    top5_probs = torch.topk(probs, 5, dim=-1).values[0].tolist()
    pred_top1 = idx_to_class[top5_indices[0]]
    pred_top5 = [idx_to_class[idx] for idx in top5_indices]
    conf = top5_probs[0]
    
    confidences.append(conf)
    cluster_stats[cluster]["conf_sum"] += conf
    
    y_true.append(true_cls)
    y_pred.append(pred_top1)
    
    is_top1 = (pred_top1.lower() == true_cls.lower())
    is_top5 = any(c.lower() == true_cls.lower() for c in pred_top5)
    
    if is_top1:
        correct_top1 += 1
        cluster_stats[cluster]["top1"] += 1
    if is_top5:
        correct_top5 += 1
        cluster_stats[cluster]["top5"] += 1
    else:
        confusions.append({"true": true_cls, "predicted": pred_top1, "confidence": round(conf, 4)})
        failed_cases.append({
            "image": p.name,
            "cluster": cluster,
            "true": true_cls,
            "predicted": pred_top1,
            "confidence": round(conf, 4),
            "top5": pred_top5,
        })

n_valid = len(y_true)
top1_acc = round((correct_top1 / n_valid) * 100.0, 2)
top5_acc = round((correct_top5 / n_valid) * 100.0, 2)
avg_conf = round(float(np.mean(confidences)), 4)
mean_lat = round(float(np.mean(latencies)), 2)

# Macro precision, recall, F1
classes_in_test = set(y_true)
prec_list = []
rec_list = []
f1_list = []

for c in classes_in_test:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)

macro_prec = round(float(np.mean(prec_list)) * 100.0, 2)
macro_rec = round(float(np.mean(rec_list)) * 100.0, 2)
macro_f1 = round(float(np.mean(f1_list)) * 100.0, 2)

# ECE Calculation
n_bins = 10
ece = 0.0
bin_counts = {}
for b in range(n_bins):
    lo = b / n_bins
    hi = (b + 1) / n_bins
    in_bin = [(c, yt == yp) for c, yt, yp in zip(confidences, y_true, y_pred) if lo <= c < hi]
    bin_counts[f"{lo:.1f}-{hi:.1f}"] = len(in_bin)
    if in_bin:
        bin_conf = np.mean([x[0] for x in in_bin])
        bin_acc = np.mean([1.0 if x[1] else 0.0 for x in in_bin])
        ece += abs(bin_conf - bin_acc) * (len(in_bin) / n_valid)

ece = round(float(ece), 4)

print(f"RESULTS FOR OPTIMIZED MODEL:")
print(f"  Top-1 Accuracy: {top1_acc}% ({correct_top1}/{n_valid})")
print(f"  Top-5 Accuracy: {top5_acc}% ({correct_top5}/{n_valid})")
print(f"  Macro Precision: {macro_prec}%")
print(f"  Macro Recall: {macro_rec}%")
print(f"  Macro F1: {macro_f1}%")
print(f"  Avg Confidence: {avg_conf}")
print(f"  ECE: {ece}")
print(f"  Latency: {mean_lat}ms")

# Backup old production checkpoint and promote candidate
backup_path = PROD_DIR / "nutrition_model_650_realworld_backup_v1.pt"
primary_path = PROD_DIR / "nutrition_model_650_realworld.pt"
best_path = PROD_DIR / "nutrition_model_650_best.pt"

if primary_path.exists():
    import shutil
    shutil.copy2(primary_path, backup_path)
    print(f"Backed up previous production model to: {backup_path.name}")

# Update production payload with calibrated final metrics
production_payload = dict(cand_ckpt)
production_payload["final_top1_accuracy"] = top1_acc
production_payload["final_top5_accuracy"] = top5_acc
production_payload["mean_confidence"] = avg_conf
production_payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

torch.save(production_payload, primary_path)
torch.save(production_payload, best_path)
print(f"Promoted optimized model to: {primary_path.name} and {best_path.name}")

# Save detailed results
detailed_report = {
    "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "total_test_images": n_valid,
    "metrics": {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "average_confidence": avg_conf,
        "ece": ece,
        "cpu_forward_latency_ms": mean_lat,
    },
    "cluster_performance": {
        k: {
            "total": v["total"],
            "top1_pct": round(v["top1"] / v["total"] * 100.0, 1),
            "top5_pct": round(v["top5"] / v["total"] * 100.0, 1),
            "avg_conf": round(v["conf_sum"] / v["total"], 4),
        }
        for k, v in cluster_stats.items()
    },
    "confidence_distribution": bin_counts,
    "confusions": confusions[:20],
    "failed_cases_count": len(failed_cases),
}

with open(PROJECT_ROOT / "scratch" / "optimized_eval_summary.json", "w", encoding="utf-8") as f:
    json.dump(detailed_report, f, indent=2)

print("Saved scratch/optimized_eval_summary.json")
