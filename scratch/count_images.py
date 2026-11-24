import os, json

classes = ['dal_makhani','shahi_paneer','vegetable_biryani','pulao','jeera_rice','khichdi','roti','naan','kulcha','paratha']
raw_root = r'C:\Nutrify\ml\data\raw'

report = {}
for cls in classes:
    dir_path = os.path.join(raw_root, cls)
    if not os.path.isdir(dir_path):
        report[cls] = {'missing': True, 'count': 0}
        continue
    files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp'))]
    report[cls] = {'missing': False, 'count': len(files)}

# Print simple lines
for cls, info in report.items():
    print(f"{cls} {info['count']}")

# Also write JSON for potential use
with open('image_counts.json','w') as jf:
    json.dump(report, jf, indent=2)
