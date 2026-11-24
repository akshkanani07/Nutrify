import os, json, sys, requests, hashlib, time
from urllib.parse import quote

# List of target classes
classes = ['dal_makhani','shahi_paneer','vegetable_biryani','pulao','jeera_rice','khichdi','roti','naan','kulcha','paratha']
raw_root = r'C:\Nutrify\ml\data\raw'

# Wikimedia API endpoint for image search
API_ENDPOINT = 'https://commons.wikimedia.org/w/api.php'

HEADERS = {'User-Agent': 'NutrifyDataCollector/0.1 (https://github.com/yourrepo)'}

def get_image_urls(class_name, limit=50):
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',
        'gsrsearch': class_name,
        'gsrnamespace': 6,  # File namespace
        'gsrlimit': limit,
        'prop': 'imageinfo',
        'iiprop': 'url',
        'iiurlwidth': 800,
        'iiurlheight': 800,
    }
    resp = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=30)
    data = resp.json()
    urls = []
    for page in data.get('query', {}).get('pages', {}).values():
        for info in page.get('imageinfo', []):
            url = info.get('url')
            if url:
                urls.append(url)
    return urls

def download_image(url, dest_path):
    try:
        r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f'Failed download {url}: {e}')
        return False

def image_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Keep a global set of hashes to avoid duplicates across classes
global_hashes = set()

for cls in classes:
    target_dir = os.path.join(raw_root, cls)
    os.makedirs(target_dir, exist_ok=True)
    existing = set(os.listdir(target_dir))
    # Count existing valid images
    count = len([f for f in existing if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp'))])
    print(f'Class {cls}: existing images {count}')
    attempts = 0
    while count < 30 and attempts < 20:
        attempts += 1
        urls = get_image_urls(cls, limit=30)
        for url in urls:
            # Simple filename from URL
            fname = os.path.basename(url.split('?')[0])
            dest = os.path.join(target_dir, fname)
            if os.path.exists(dest):
                continue
            # Download to temp path
            temp_path = dest + '.tmp'
            if not download_image(url, temp_path):
                continue
            # Basic validation: ensure file not empty and reasonable size (>10KB)
            if os.path.getsize(temp_path) < 10*1024:
                os.remove(temp_path)
                continue
            # Deduplication via hash
            h = image_hash(temp_path)
            if h in global_hashes:
                os.remove(temp_path)
                continue
            global_hashes.add(h)
            os.rename(temp_path, dest)
            count += 1
            if count >= 30:
                break
        print(f'Class {cls}: downloaded up to {count} images after attempt {attempts}')
    print(f'Finished class {cls} with {count} images')

print('All classes processed.')
