import os, time, json

raw_root = r'C:\\Nutrify\\ml\\data\\raw'
classes = ['dal_makhani','shahi_paneer','vegetable_biryani','jeera_rice','roti']

def count_images(cls):
    dir_path = os.path.join(raw_root, cls)
    if not os.path.isdir(dir_path):
        return 0
    return len([f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg','.jpeg','.png','.webp','.bmp'))])

def all_ready():
    for cls in classes:
        if count_images(cls) < 30:
            return False
    return True

print('Waiting for missing images to be added...')
while not all_ready():
    missing = {cls: 30 - count_images(cls) for cls in classes if count_images(cls) < 30}
    print('Still missing:', missing)
    time.sleep(30)  # check every 30 seconds

print('All required images present. Proceeding.')
# Write a simple JSON flag for downstream steps
with open('C:\\Nutrify\\scratch\\images_ready_flag.json', 'w') as f:
    json.dump({'ready': True}, f)
