import os, requests, hashlib, time

raw_root = r'C:\\Nutrify\\ml\\data\\raw'
# Deficits
needs = {
    'dal_makhani': 4,
    'shahi_paneer': 4,
    'vegetable_biryani': 2,
    'jeera_rice': 1,
    'roti': 1,
}
# Unsplash source base URL (no API key needed)
BASE_URL = 'https://source.unsplash.com/800x800/?'
HEADERS = {'User-Agent': 'NutrifyDataCollector/0.4 (https://github.com/yourrepo)'}

def download_unsplash(query, dest_path):
    url = BASE_URL + query.replace(' ', '%20')
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f'Unsplash download error for {query}: {e}')
        return False

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Build global hash set
global_hashes = set()
for cls in os.listdir(raw_root):
    d = os.path.join(raw_root, cls)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp')):
                try:
                    global_hashes.add(file_hash(os.path.join(d, f)))
                except Exception:
                    pass

for cls, needed in needs.items():
    target_dir = os.path.join(raw_root, cls)
    os.makedirs(target_dir, exist_ok=True)
    attempts = 0
    while needed > 0 and attempts < 100:
        attempts += 1
        query = cls.replace('_', ' ')  # use class name as query
        fname = f'unsplash_{int(time.time()*1000)}_{attempts}.jpg'
        dest = os.path.join(target_dir, fname)
        if download_unsplash(query, dest):
            # size check
            if os.path.getsize(dest) < 10*1024:
                os.remove(dest)
                continue
            h = file_hash(dest)
            if h in global_hashes:
                os.remove(dest)
                continue
            global_hashes.add(h)
            needed -= 1
            print(f"{cls}: added one, {needed} remaining")
        else:
            # failed download, break maybe
            continue
    print(f"Finished {cls}, remaining need {needed}")
print('All classes processed')
