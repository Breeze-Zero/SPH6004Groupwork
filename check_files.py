import os
from concurrent.futures import ThreadPoolExecutor

with open('/home/e1373616/project/sph6004/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES', 'r') as f:
    lines = [line.strip() for line in f]

base_dir = '/home/e1373616/project/sph6004/physionet.org/files/mimic-cxr-jpg/2.1.0'
def check_file(file_name):
    return os.path.exists(os.path.join(base_dir, file_name))

with ThreadPoolExecutor(max_workers=32) as executor:
    results = list(executor.map(check_file, lines))

n = sum(results)
print(f'Download {n}, {round(n/len(lines), 4)*100}%')
