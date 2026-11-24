import os, requests, hashlib

raw_root = r'C:\\Nutrify\\ml\\data\\raw'
# Classes that still need images (as of latest count)
needs = {
    'dal_makhani': 4,
    'shahi_paneer': 4,
    'vegetable_biryani': 2,
    'jeera_rice': 1,
    'roti': 1,
}
API_ENDPOINT = 'https://commons.wikimedia.org/w/api.php'
HEADERS = {'User-Agent': 'NutrifyDataCollector/0.2 (https://github.com/yourrepo)'}

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
    r = requests.get(API_ENDPOINT, params=params, headers=HEADERS, timeout=30)
    data = r.json()
    urls = []
    for page in data.get('query', {}).get('pages', {}).values():
        for info in page.get('imageinfo', []):
            url = info.get('url')
            if url:
                urls.append(url)
    return urls

def download(url, dest):
    r = requests.get(url, headers=HEADERS, stream=True, timeout=20)
    r.raise_for_status()
    with open(dest, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# Build global hash set to avoid duplicates across classes
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
    while needed > 0 and attempts < 30:
        attempts += 1
        urls = get_image_urls(cls, limit=30)
        for url in urls:
            fname = os.path.basename(url.split('?')[0])
            dest = os.path.join(target_dir, fname)
            if os.path.exists(dest):
                continue
            tmp = dest + '.tmp'
            try:
                download(url, tmp)
            except Exception:
                continue
            # size check
            if os.path.getsize(tmp) < 10*1024:
                os.remove(tmp)
                continue
            h = file_hash(tmp)
            if h in global_hashes:
                os.remove(tmp)
                continue
            global_hashes.add(h)
            os.rename(tmp, dest)
            needed -= 1
            if needed == 0:
                break
        print(f"{cls}: still need {needed} after attempt {attempts}")
    print(f"Finished {cls}, added images, remaining need {needed}")
print('All target classes processed')
