import os, sys, requests, time, hashlib
from PIL import Image

sys.stdout.reconfigure(line_buffering=True)

raw_root = r'C:\Nutrify\ml\data\raw'
needs = {
    'dal_makhani': 4,
    'shahi_paneer': 4,
    'vegetable_biryani': 2,
    'jeera_rice': 1,
    'roti': 1
}

sources = {
    'dal_makhani': [
        ('cat', 'Dal makhani'),
        ('search', 'dal makhani food'),
        ('search', 'dal makhni curry'),
        ('search', 'maa ki dal'),
        ('search', 'black lentils dal'),
    ],
    'shahi_paneer': [
        ('cat', 'Shahi paneer'),
        ('search', 'shahi paneer food'),
        ('search', 'paneer butter masala dish'),
        ('search', 'paneer makhani'),
        ('search', 'royal paneer'),
    ],
    'vegetable_biryani': [
        ('search', 'vegetable biryani dish'),
        ('search', 'veg biryani rice'),
        ('search', 'hyderabadi veg biryani'),
        ('search', 'dum vegetable biryani'),
    ],
    'jeera_rice': [
        ('search', 'jeera rice food'),
        ('search', 'cumin rice indian'),
        ('search', 'zeera rice dish'),
        ('search', 'jeera chawal food'),
    ],
    'roti': [
        ('cat', 'Chapati'),
        ('search', 'phulka chapati'),
        ('search', 'tawa roti indian flatbread'),
        ('search', 'roti indian bread food'),
    ]
}

headers = {'User-Agent': 'NutrifyDatasetAcquisition/1.0 (FoodAI Research; contact: aksh@example.com)'}

def dhash(image, hash_size=8):
    # Difference hash for visual similarity
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    difference = []
    for row in range(hash_size):
        for col in range(hash_size):
            pixel_left = pixels[row * (hash_size + 1) + col]
            pixel_right = pixels[row * (hash_size + 1) + col + 1]
            difference.append(pixel_left > pixel_right)
    decimal_value = 0
    hex_string = []
    for index, value in enumerate(difference):
        if value:
            decimal_value += 2**(index % 8)
        if (index % 8) == 7:
            hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
            decimal_value = 0
    return ''.join(hex_string)

def hamming_dist(s1, s2):
    return bin(int(s1, 16) ^ int(s2, 16)).count('1')

def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(16384)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

print("Indexing existing images and perceptual hashes...", flush=True)
existing_sha256 = set()
existing_dhashes = {}  # class -> list of dhashes

for root, dirs, files in os.walk(raw_root):
    cls_name = os.path.basename(root)
    existing_dhashes[cls_name] = []
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            full_p = os.path.join(root, f)
            try:
                sha = get_file_hash(full_p)
                existing_sha256.add(sha)
                with Image.open(full_p) as img:
                    dh = dhash(img)
                    existing_dhashes[cls_name].append(dh)
            except Exception:
                pass

print(f"Total existing SHA-256 indexed: {len(existing_sha256)}", flush=True)

def query_commons(stype, val):
    url = 'https://commons.wikimedia.org/w/api.php'
    if stype == 'cat':
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'categorymembers',
            'gcmtitle': f'Category:{val}',
            'gcmlimit': 50,
            'prop': 'imageinfo',
            'iiprop': 'url|size|mime',
            'iiurlwidth': 800
        }
    else:
        params = {
            'action': 'query',
            'format': 'json',
            'generator': 'search',
            'gsrsearch': val,
            'gsrnamespace': 6,
            'gsrlimit': 50,
            'prop': 'imageinfo',
            'iiprop': 'url|size|mime',
            'iiurlwidth': 800
        }
    try:
        time.sleep(0.5)
        r = requests.get(url, params=params, headers=headers, timeout=20)
        data = r.json()
        items = []
        for p in data.get('query', {}).get('pages', {}).values():
            if 'imageinfo' in p:
                ii = p['imageinfo'][0]
                turl = ii.get('thumburl') or ii.get('url')
                items.append({
                    'title': p.get('title'),
                    'url': turl,
                    'orig_url': ii.get('url'),
                    'width': ii.get('thumbwidth', ii.get('width', 0)),
                    'height': ii.get('thumbheight', ii.get('height', 0)),
                    'mime': ii.get('mime')
                })
        return items
    except Exception as e:
        print(f"Error querying {stype}:{val}: {e}", flush=True)
        return []

collection_results = {}

for cls, needed in needs.items():
    target_dir = os.path.join(raw_root, cls)
    os.makedirs(target_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))]
    prev_count = len(existing_files)
    target_count = 30
    to_add = max(0, target_count - prev_count)
    
    print(f"\n==========================================", flush=True)
    print(f"Class: {cls} | Current: {prev_count} | Needed: {to_add}", flush=True)
    print(f"==========================================", flush=True)
    
    added_for_cls = []
    
    if to_add == 0:
        collection_results[cls] = {'prev': prev_count, 'added': 0, 'final': prev_count, 'items': []}
        continue
        
    candidates = []
    for stype, val in sources[cls]:
        c_list = query_commons(stype, val)
        candidates.extend(c_list)
        if len(candidates) >= 60:
            break
            
    print(f"Retrieved {len(candidates)} candidate images from Wikimedia", flush=True)
    
    for cand in candidates:
        if len(added_for_cls) >= to_add:
            break
            
        turl = cand['url']
        title = cand['title']
        if not turl:
            continue
        if cand['mime'] and not cand['mime'].startswith('image/'):
            continue
        if turl.lower().endswith(('.svg', '.gif')):
            continue
            
        # Avoid non-food titles
        bad_keywords = ['map', 'flag', 'logo', 'diagram', 'chart', 'mountain', 'weissenstein', 'hooned']
        if any(b in title.lower() for b in bad_keywords):
            continue
            
        temp_dest = os.path.join(r"C:\Nutrify\scratch", f"cand_{int(time.time()*1000)}.jpg")
        try:
            time.sleep(0.3)
            r = requests.get(turl, headers=headers, timeout=20)
            if r.status_code != 200:
                continue
            with open(temp_dest, 'wb') as f:
                f.write(r.content)
                
            if os.path.getsize(temp_dest) < 15000:
                os.remove(temp_dest)
                continue
                
            # Verify image with PIL
            with Image.open(temp_dest) as img:
                img.verify()
                
            with Image.open(temp_dest) as img:
                w, h = img.size
                if w < 224 or h < 224:
                    os.remove(temp_dest)
                    continue
                dh = dhash(img)
                
            sha = get_file_hash(temp_dest)
            if sha in existing_sha256:
                os.remove(temp_dest)
                continue
                
            # Check near-duplicate in same class
            is_near_dup = False
            for edh in existing_dhashes.get(cls, []):
                if hamming_dist(dh, edh) < 6:  # very close visually
                    is_near_dup = True
                    break
            if is_near_dup:
                os.remove(temp_dest)
                continue
                
            # Valid image! Save to raw class folder
            final_name = f"real_{cls}_{len(added_for_cls)+1}_{int(time.time())}.jpg"
            final_path = os.path.join(target_dir, final_name)
            os.rename(temp_dest, final_path)
            
            existing_sha256.add(sha)
            existing_dhashes[cls].append(dh)
            added_for_cls.append({
                'filename': final_name,
                'source_title': title,
                'width': w,
                'height': h,
                'size_kb': round(os.path.getsize(final_path)/1024, 1),
                'sha256': sha
            })
            print(f"  [+] SAVED: {final_name} ({w}x{h}, {round(os.path.getsize(final_path)/1024,1)}KB) from {title}", flush=True)
            
        except Exception as e:
            if os.path.exists(temp_dest):
                try: os.remove(temp_dest)
                except Exception: pass
            print(f"  [-] Failed processing {title}: {e}", flush=True)
            
    final_count = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))])
    collection_results[cls] = {
        'prev': prev_count,
        'added': len(added_for_cls),
        'final': final_count,
        'items': added_for_cls
    }
    print(f"Class {cls} COMPLETE: prev={prev_count}, added={len(added_for_cls)}, final={final_count}", flush=True)

print("\n==========================================", flush=True)
print("FINAL ACQUISITION SUMMARY", flush=True)
print("==========================================", flush=True)
all_satisfied = True
total_added = 0
for cls, res in collection_results.items():
    print(f"{cls:20} | Previous: {res['prev']:2} | Added: {res['added']:2} | Final: {res['final']:2} | Target: 30")
    total_added += res['added']
    if res['final'] < 30:
        all_satisfied = False

print(f"Total new images added: {total_added}")
print(f"All classes meet >= 30: {all_satisfied}")

# If all satisfied, write flag file if not already written
if all_satisfied:
    flag_file = r'C:\Nutrify\scratch\images_ready_flag.json'
    import json
    with open(flag_file, 'w') as f:
        json.dump({'ready': True, 'total_added': total_added, 'results': collection_results}, f, indent=2)
    print(f"Flag file created at: {flag_file}")
