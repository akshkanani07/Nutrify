import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "backend")
from backend.app.providers.ml_provider import MLModelProvider

provider = MLModelProvider()
print("Model loaded successfully. Classes:", provider._service.num_classes)
print("Checkpoint:", provider.checkpoint_path)

user_dir = Path(r"C:\Users\AkshKanani\.gemini\antigravity\brain\b7d6aba7-8151-4941-a38f-feefc1012f2a\.user_uploaded")
for img_path in sorted(user_dir.glob("*.*")):
    try:
        res = provider._service.predict(img_path, top_k=5)
        print(f"\n{img_path.name}:")
        print(f"  Top-1: {res['food_name']} ({res['confidence']*100:.2f}%)")
        print(f"  Candidates: {[(c['food_name'], round(c['confidence']*100, 2)) for c in res['candidates']]}")
    except Exception as e:
        print(f"\n{img_path.name}: ERROR {e}")

