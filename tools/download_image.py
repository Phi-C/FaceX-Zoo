from concurrent import futures
import os
import requests
import tqdm
import math
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_one_image(url, file_name):
    '''
    Given url of an image, download and save.
    Args:
        url: universal resource location of image
        file_name: image path
    '''
    if os.path.isfile(file_name):
        return

    try:
        re = requests.get(url, stream=True, timeout=5)
        if re.status_code == 200:
            with open(file_name, 'wb') as f:
                f.write(re.content)
        del re
    except:
        print("Failed to download {}".format(url))

def download_images(src_image_pathes, dst_image_pathes, pb):
    assert(len(src_image_pathes) == len(dst_image_pathes))

    for i in range(len(src_image_pathes)):
        download_one_image(src_image_pathes[i], dst_image_pathes[i])
        if pb is not None:
            pb.update(1)

def parallel_download_images(url_file_name, dst_root):
    '''
    Speed up, use multi threads to download images.
    Args:
        url_file_name: a .txt file, each line contains a url for an image
        dst_root: directory path for downloaded images 
    '''
    with open(url_file_name, 'r') as f:
        data = f.readlines()
    
    src_pathes = []
    dst_pathes = []
    for idx, item in enumerate(data):
        src_pathes.append(item.strip())
        dst_pathes.append(os.path.join(dst_root, "{:05d}.jpg".format(idx)))
    
    workers = 10
    work_load_total = len(data)
    work_load_per_thread = math.ceil(work_load_total / workers)
    pb = tqdm.tqdm(total=work_load_total, desc="Downloading")

    futures_for_thread = dict()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for tid in range(workers):
            st = tid * work_load_per_thread
            ed = tid * work_load_per_thread + work_load_per_thread
            if st > work_load_total:
                break
            if ed > work_load_total:
                ed = work_load_total

            futures_for_thread[executor.submit(download_images, src_pathes[st:ed], dst_pathes[st:ed], pb)] = tid

    for future in as_completed(futures_for_thread):
        tid = futures_for_thread[future]
        data = future.result()

    pb.close()


if __name__ == "__main__":
    url = ''
    file_name = 'test.jpg'
    download_one_image(url, file_name)