import os, sys, requests, time, hashlib
from PIL import Image

raw_root = r'C:\Nutrify\ml\data\raw'
needs = {
    'dal_makhani': 4,
    'shahi_paneer': 4,
    'vegetable_biryani': 2,
    'jeera_rice': 1,
    'roti': 1
}

# Queries and categories to check for each class
sources = {
    'dal_makhani': [
        ('cat', 'Dal makhani'),
        ('search', 'dal makhani'),
        ('search', 'dal makhni'),
        ('search', 'maa ki dal'),
        ('search', 'black lentil dal'),
    ],
    'shahi_paneer': [
        ('cat', 'Shahi paneer'),
        ('search', 'shahi paneer'),
        ('search', 'paneer butter masala'),
        ('search', 'paneer makhani'),
        ('search', 'paneer gravy'),
    ],
    'vegetable_biryani': [
        ('search', 'vegetable biryani'),
        ('search', 'veg biryani'),
        ('search', 'biryani veg'),
        ('search', 'vegetable dum biryani'),
        ('search', 'veg dum biryani'),
    ],
    'jeera_rice': [
        ('search', 'jeera rice'),
        ('search', 'cumin rice'),
        ('search', 'zeera rice'),
        ('search', 'jeera chawal'),
        ('search', 'jeera pulao'),
    ],
    'roti': [
        ('cat', 'Chapati'),
        ('cat', 'Roti'),
        ('search', 'chapati flatbread'),
        ('search', 'phulka roti'),
        ('search', 'tawa roti'),
        ('search', 'tandoori roti'),
    ]
}

headers = {'User-Agent': 'NutrifyDatasetAcquisition/1.0 (FoodRecognitionResearch; contact: aksh@example.com)'}

def get_cat_members(cat):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'categorymembers',
        'gcmtitle': f'Category:{cat}',
        'gcmlimit': 50,
        'prop': 'imageinfo',
        'iiprop': 'url|size|mime'
    }
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        data = r.json()
        items = []
        for p in data.get('query', {}).get('pages', {}).values():
            if 'imageinfo' in p:
                ii = p['imageinfo'][0]
                items.append((p.get('title'), ii.get('url'), ii.get('width', 0), ii.get('height', 0), ii.get('mime')))
        return items
    except Exception as e:
        print(f"Error fetching category {cat}: {e}")
        return []

def get_search_results(q):
    url = 'https://commons.wikimedia.org/w/api.php'
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',
        'gsrsearch': q,
        'gsrnamespace': 6,
        'gsrlimit': 50,
        'prop': 'imageinfo',
        'iiprop': 'url|size|mime'
    }
    try:
        time.sleep(0.5)
        r = requests.get(url, params=params, headers=headers, timeout=20)
        data = r.json()
        items = []
        for p in data.get('query', {}).get('pages', {}).values():
            if 'imageinfo' in p:
                ii = p['imageinfo'][0]
                items.append((p.get('title'), ii.get('url'), ii.get('width', 0), ii.get('height', 0), ii.get('mime')))
        return items
    except Exception as e:
        print(f"Error searching {q}: {e}")
        return []

def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(16384)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Collect all existing hashes across raw directory
print("Indexing existing images...")
existing_hashes = set()
existing_names = set()
for root, dirs, files in os.walk(raw_root):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            full_p = os.path.join(root, f)
            existing_names.add(f.lower())
            try:
                h = get_file_hash(full_p)
                existing_hashes.add(h)
            except Exception:
                pass

print(f"Total existing indexed hashes: {len(existing_hashes)}")

added_summary = {}

for cls, needed in needs.items():
    target_dir = os.path.join(raw_root, cls)
    os.makedirs(target_dir, exist_ok=True)
    existing_in_class = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
    target_count = 30
    to_add = max(0, target_count - existing_in_class)
    print(f"\nProcessing {cls}: current={existing_in_class}, target={target_count}, needed={to_add}")
    
    added_for_cls = []
    
    if to_add == 0:
        added_summary[cls] = {'prev': existing_in_class, 'added': 0, 'final': existing_in_class}
        continue
        
    candidates = []
    for stype, query in sources[cls]:
        if stype == 'cat':
            items = get_cat_members(query)
        else:
            items = get_search_results(query)
        candidates.extend(items)
        if len(candidates) > 50:
            break
            
    print(f"Found {len(candidates)} candidates for {cls}")
    
    for title, url, width, height, mime in candidates:
        if len(added_for_cls) >= to_add:
            break
        if not url:
            continue
        # Skip SVG or non-image
        if mime and not mime.startswith('image/'):
            continue
        if url.lower().endswith('.svg'):
            continue
        if width < 224 or height < 224:
            continue
            
        fname = os.path.basename(url.split('?')[0])
        # Make a safe filename
        safe_fname = f"real_{cls}_{len(added_for_cls)+1}_{int(time.time())}.jpg"
        temp_dest = os.path.join(r"C:\Nutrify\scratch", f"tmp_{safe_fname}")
        
        try:
            time.sleep(0.3)
            r = requests.get(url, headers=headers, timeout=25, stream=True)
            if r.status_code != 200:
                continue
            with open(temp_dest, 'wb') as tf:
                for chunk in r.iter_content(8192):
                    tf.write(chunk)
            
            # Validate with PIL
            with Image.open(temp_dest) as img:
                img.verify()
            with Image.open(temp_dest) as img:
                w, h = img.size
                fmt = img.format
                if w < 224 or h < 224:
                    os.remove(temp_dest)
                    continue
                # Reject if tiny file size < 15KB
                if os.path.getsize(temp_dest) < 15000:
                    os.remove(temp_dest)
                    continue
                    
            img_hash = get_file_hash(temp_dest)
            if img_hash in existing_hashes:
                os.remove(temp_dest)
                continue
                
            final_dest = os.path.join(target_dir, safe_fname)
            os.rename(temp_dest, final_dest)
            existing_hashes.add(img_hash)
            added_for_cls.append(safe_fname)
            print(f"  [+] Added {safe_fname} ({w}x{h}, {fmt}) from {title}")
        except Exception as e:
            if os.path.exists(temp_dest):
                try:
                    os.remove(temp_dest)
                except Exception:
                    pass
            print(f"  [-] Failed for {title}: {e}")
            continue
            
    final_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
    added_summary[cls] = {'prev': existing_in_class, 'added': len(added_for_cls), 'final': final_count}
    print(f"Class {cls} final count: {final_count} (added {len(added_for_cls)})")

print("\n=== SUMMARY ===")
for cls, info in added_summary.items():
    print(f"{cls}: previous={info['prev']}, added={info['added']}, final={info['final']}")
