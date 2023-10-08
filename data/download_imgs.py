import argparse
import json
import os
import time

import requests
import pandas as pd
import glob
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from pathlib import Path
from requests.exceptions import MissingSchema, Timeout, ConnectionError, InvalidSchema


def download_one(entry, overwrite=False):
    fn, uri, target_pth, retries = entry
    fn = fn.replace("/", "_")
    path = f'{target_pth}/{fn}'
    if os.path.exists(path) and not overwrite:
        return fn, None

    for i in range(retries):
        try:
            r = requests.get(uri, stream=True, timeout=50)
        except (MissingSchema, Timeout, ConnectionError, InvalidSchema):
            time.sleep(i)
            continue

        if r.status_code == 200:
            with open(path, 'wb') as f:
                for chunk in r:
                    f.write(chunk)
            return fn, None
        else:
            time.sleep(i)
            continue
    try:
        Path(f'{target_pth}/failed/{fn}').touch()
    except:
        pass

    return None, fn


def download_all(metadata_pth, target_pth, retries):
    df = pd.read_csv(metadata_pth)
    entries = [[*x, target_pth, retries] for x in df[['File Name', 'Image Credits']].values]
    os.makedirs(f'{target_pth}/failed', exist_ok=True)
    n_processes = max(1, cpu_count() - 1)
    with ThreadPool(n_processes) as p:
        results = list(tqdm(p.imap(download_one, entries), total=len(entries)))
    return results


def filter_coco(coco_pth, target_pth, imgs_folder):
    """Filter coco json to only contain images present in the images folder"""
    with open(coco_pth) as f:
        coco = json.load(f)
    fns_present = list(map(lambda p: os.path.basename(p), glob.glob(f'{imgs_folder}/*.jpg')))
    n_remove = len(coco['images']) - len(fns_present)
    imgs_filtered = [img for img in coco['images'] if img['file_name'] in fns_present]
    img_ids_filtered = [img['id'] for img in imgs_filtered]
    anns_filtered = [ann for ann in coco['annotations'] if ann['image_id'] in img_ids_filtered]
    coco['images'] = imgs_filtered
    coco['annotations'] = anns_filtered
    # ignore categories for now
    with open(target_pth, 'w') as f:
        json.dump(coco, f)
    print(f'Filtered annotations saved to {coco_target} with {n_remove} images removed.')


def dump_failing_records(failing_dls, metadata_pth, records_target):
    df = pd.read_csv(metadata_pth)
    failing_records = df[df['File Name'].isin(failing_dls)]
    failing_records.to_csv(records_target)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Download images for given coco json and metadata records.')
    parser.add_argument('--csv', type=str, help='Path to metadata.csv with file names and download links', default='meta.csv', required=False)
    parser.add_argument('--coco', type=str, help='Path to coco json with images array to download.', default='instances_all.json', required=False)
    parser.add_argument('--imgs', type=str, help='Path to directory to download the images to.', default='imgs',
                        required=False)
    parser.add_argument('--retries', type=str, help='Number of times to retry download.', default=2)
    parser.add_argument('--records_target', type=str, help='Path to records file to keep track of failing dls', default='./failing_dls.csv')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    results = download_all(args.csv, args.imgs, args.retries)
    print('Download complete.')
    failing_dls = [fn for _, fn in results if fn is not None]
    if args.coco:
        coco_target = f'{os.path.splitext(args.coco)[0]}_downloadable.json'
        filter_coco(args.coco, coco_target, args.imgs)
    dump_failing_records(failing_dls, args.csv, args.records_target)
    print(f'Records with corrupt download links saved to {args.records_target}.')
