import os, json, hashlib, requests

classes = ['dal_makhani','shahi_paneer','vegetable_biryani','pulao','jeera_rice','khichdi','roti','naan','kulcha','paratha']
raw_root = r'C:\Nutrify\ml\data\raw'

API_ENDPOINT = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'NutrifyDataCollector/0.1 (https://github.com/yourrepo)'}

def get_image_urls(class_name, limit=50):
    params = {
        'action': 'query',
        'format': 'json',
        'generator': 'search',
        'gsrsearch': class_name,
        'gsrnamespace': 6,
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

# Build set of existing hashes across all classes
existing_hashes = set()
for cls in classes:
    dir_path = os.path.join(raw_root, cls)
    if not os.path.isdir(dir_path):
        continue
    for f in os.listdir(dir_path):
        if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp')):
            full = os.path.join(dir_path, f)
            try:
                existing_hashes.add(image_hash(full))
            except Exception:
                pass

for cls in classes:
    target_dir = os.path.join(raw_root, cls)
    os.makedirs(target_dir, exist_ok=True)
    current = len([f for f in os.listdir(target_dir) if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp'))])
    needed = 30 - current
    if needed <= 0:
        print(f'{cls} already has {current} images')
        continue
    print(f'Filling {cls}: need {needed} more images')
    attempts = 0
    while needed > 0 and attempts < 30:
        attempts += 1
        urls = get_image_urls(cls, limit=30)
        for url in urls:
            fname = os.path.basename(url.split('?')[0])
            dest = os.path.join(target_dir, fname)
            if os.path.exists(dest):
                continue
            temp = dest + '.tmp'
            if not download_image(url, temp):
                continue
            if os.path.getsize(temp) < 10*1024:
                os.remove(temp)
                continue
            h = image_hash(temp)
            if h in existing_hashes:
                os.remove(temp)
                continue
            existing_hashes.add(h)
            os.rename(temp, dest)
            needed -= 1
            if needed == 0:
                break
        print(f'After attempt {attempts}, {cls} still needs {needed}')
    print(f'Finished {cls} with {30 - needed} new images (total now {30 - needed + current})')
print('All missing images filled')
