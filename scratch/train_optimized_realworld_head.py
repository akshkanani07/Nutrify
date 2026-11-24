"""
Nutrify AI — Final Model Optimization & Feature Separation Experiment
Optimizes representation learning for hard-negative categories and tests candidate checkpoint.
"""
import os, sys, json, time
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchvision import models, transforms

PROJECT_ROOT = Path(r"C:\Nutrify")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.scalable_trainer import ScalableClassifierHead, ScalableFoodModel, build_backbone
from ml.src.metrics import calculate_topk_accuracy

PROD_DIR = PROJECT_ROOT / "ml" / "models" / "production"
CATALOG_PATH = PROJECT_ROOT / "ml" / "knowledge_base" / "expanded_catalog_650.json"
REAL_WORLD_DIR = PROJECT_ROOT / "ml" / "data" / "real_world_dataset"
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw"

# Maximize CPU throughput
threads = min(8, os.cpu_count() or 4)
torch.set_num_threads(threads)
device = torch.device("cpu")
print(f"Using device: {device} with {threads} CPU threads")

with open(CATALOG_PATH, "r", encoding="utf-8") as f:
    catalog = json.load(f)

all_classes = sorted(list(catalog["classes"].keys()))
num_classes = len(all_classes)
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for i, c in enumerate(all_classes)}

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

# Image transforms with realistic food augmentations
train_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
]

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Build base model and load base weights
extractor, in_features, _ = build_backbone("efficientnet_b0", pretrained=False)
head = ScalableClassifierHead(in_features=in_features, num_classes=num_classes)
model = ScalableFoodModel(extractor, head, "efficientnet_b0")

base_ckpt_path = PROD_DIR / "nutrition_model_650_best.pt"
if base_ckpt_path.exists():
    print(f"Loading base checkpoint: {base_ckpt_path}")
    base_payload = torch.load(base_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(base_payload["model_state_dict"])
else:
    print(f"Error: {base_ckpt_path} not found")
    sys.exit(1)

model.eval()

# Collect dataset:
# For priority classes: training images 0 to 23 from raw/
# For other 650 classes: images from real_world_dataset/
train_images = []
val_images = []

# 1. Collect from real_world_dataset for all 650 classes
for cname in all_classes:
    c_idx = class_to_idx[cname]
    folder = REAL_WORLD_DIR / cname.replace(" ", "_")
    if folder.exists():
        imgs = sorted(list(folder.glob("*.jpg")))
        if len(imgs) >= 3:
            train_images.append((imgs[0], c_idx))
            train_images.append((imgs[1], c_idx))
            val_images.append((imgs[2], c_idx))
        elif len(imgs) == 2:
            train_images.append((imgs[0], c_idx))
            val_images.append((imgs[1], c_idx))
        elif len(imgs) == 1:
            train_images.append((imgs[0], c_idx))
            val_images.append((imgs[0], c_idx))

# 2. Add authentic priority training images (indices 0..23)
for raw_cls, target_cls in CLASS_MAPPING.items():
    raw_folder = RAW_DIR / raw_cls
    if raw_folder.exists():
        imgs = sorted([p for p in raw_folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}])
        train_slice = imgs[:24]  # First 24 are training
        target_idx = class_to_idx[target_cls]
        for p in train_slice:
            train_images.append((p, target_idx))

print(f"Total training image paths: {len(train_images)}, validation paths: {len(val_images)}")

# Extract features with augmentation
extractor.eval()
Image.MAX_IMAGE_PIXELS = 25_000_000

print("Extracting features with data augmentation for training...")
t0 = time.time()
feats_list = []
labels_list = []

with torch.no_grad():
    for idx, (img_path, target_idx) in enumerate(train_images):
        try:
            with Image.open(img_path) as raw_img:
                rgb_img = raw_img.convert("RGB")
                # For priority classes, use all 3 transforms; for others, use base transform
                is_priority = any(raw_cls in str(img_path) for raw_cls in CLASS_MAPPING)
                active_transforms = train_transforms if is_priority else [train_transforms[0]]
                
                for t in active_transforms:
                    tensor = t(rgb_img).unsqueeze(0).to(device)
                    f = extractor(tensor)
                    feats_list.append(f)
                    labels_list.append(target_idx)
        except Exception as e:
            continue
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(train_images):
            print(f"  Processed {idx + 1}/{len(train_images)} training images...")

train_feats = torch.cat(feats_list, dim=0)
train_labels = torch.tensor(labels_list, dtype=torch.long)
print(f"Training feature set created: {train_feats.shape} in {time.time() - t0:.1f}s")

# Extract validation features
val_feats_list = []
val_labels_list = []
with torch.no_grad():
    for img_path, target_idx in val_images:
        try:
            with Image.open(img_path) as raw_img:
                tensor = val_transform(raw_img.convert("RGB")).unsqueeze(0).to(device)
                f = extractor(tensor)
                val_feats_list.append(f)
                val_labels_list.append(target_idx)
        except Exception:
            continue

val_feats = torch.cat(val_feats_list, dim=0)
val_labels = torch.tensor(val_labels_list, dtype=torch.long)
print(f"Validation feature set created: {val_feats.shape}")

# Train the candidate classifier head
head = ScalableClassifierHead(in_features=in_features, num_classes=num_classes)
head.to(device)
head.train()

optimizer = AdamW(head.parameters(), lr=1.5e-3, weight_decay=2e-4)
epochs = 45
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.08)

dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

print("Training candidate classifier head...")
best_val_top1 = 0.0
best_state = None

for ep in range(1, epochs + 1):
    head.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for bx, by in loader:
        optimizer.zero_grad()
        logits = head(bx)
        loss = criterion(logits, by)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * bx.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == by).sum().item()
        total += bx.size(0)
        
    scheduler.step()
    train_acc = (correct / total) * 100.0
    avg_loss = total_loss / total
    
    head.eval()
    with torch.no_grad():
        v_logits = head(val_feats)
        v_top1, v_top5 = calculate_topk_accuracy(v_logits, val_labels, topk=(1, 5))
        
    if v_top1 > best_val_top1:
        best_val_top1 = v_top1
        best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        
    if ep % 5 == 0 or ep == epochs:
        print(f"Epoch [{ep:2d}/{epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:5.2f}% | Val Top-1: {v_top1:5.2f}% | Val Top-5: {v_top5:5.2f}%")

# Save candidate model
if best_state is not None:
    head.load_state_dict(best_state)

model.head = head
candidate_ckpt_path = PROD_DIR / "candidate_model_650_realworld.pt"
candidate_payload = {
    "model_name": "efficientnet_b0",
    "num_classes": num_classes,
    "class_to_idx": class_to_idx,
    "idx_to_class": idx_to_class,
    "model_state_dict": model.state_dict(),
    "final_top1_accuracy": round(best_val_top1, 2),
    "input_resolution": [224, 224],
    "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
torch.save(candidate_payload, candidate_ckpt_path)
print(f"Saved candidate checkpoint to: {candidate_ckpt_path}")

# =====================================================================
# EVALUATE CANDIDATE ON EXACT SAME 138 UNSEEN REAL-WORLD TEST SET
# =====================================================================
print("\n" + "=" * 60)
print("EVALUATING CANDIDATE ON UNSEEN AUTHENTIC TEST SET (138 images)")
print("=" * 60)

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

model.eval()
correct_top1 = 0
correct_top5 = 0
valid_cases = 0

with torch.no_grad():
    for p, true_cls, cluster in test_cases:
        try:
            with Image.open(p) as img:
                t = val_transform(img.convert("RGB")).unsqueeze(0).to(device)
        except Exception:
            continue
        
        logits = model(t)
        probs = F.softmax(logits, dim=-1)
        top5_indices = torch.topk(probs, 5, dim=-1).indices[0].tolist()
        pred_top1 = idx_to_class[top5_indices[0]]
        pred_top5 = [idx_to_class[idx] for idx in top5_indices]
        
        valid_cases += 1
        if pred_top1.lower() == true_cls.lower():
            correct_top1 += 1
        if any(c.lower() == true_cls.lower() for c in pred_top5):
            correct_top5 += 1

cand_top1 = round(correct_top1 / valid_cases * 100, 2)
cand_top5 = round(correct_top5 / valid_cases * 100, 2)

print(f"Evaluated on {valid_cases} unseen real-world images:")
print(f"  Candidate Model -> Top-1: {cand_top1}% ({correct_top1}/{valid_cases}), Top-5: {cand_top5}% ({correct_top5}/{valid_cases})")
print(f"  Current Model   -> Top-1: 5.80% (8/138), Top-5: 12.32% (17/138)")

# Save evaluation comparison to a json log
eval_res = {
    "candidate_top1": cand_top1,
    "candidate_top5": cand_top5,
    "current_top1": 5.80,
    "current_top5": 12.32,
    "is_candidate_better": cand_top1 >= 5.80 and cand_top5 >= 12.32,
}
with open(PROJECT_ROOT / "scratch" / "candidate_eval_results.json", "w") as f:
    json.dump(eval_res, f, indent=2)

if eval_res["is_candidate_better"]:
    print("\n✓ CANDIDATE MODEL IS SUPERIOR OR EQUAL!")
else:
    print("\n⚠ Current production model remains superior or equal. Preserving current production weights.")
