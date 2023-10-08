import itertools
import json

from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts, DatasetCatalog, MetadataCatalog,
)
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode

from detrex.data import DetrDatasetMapper

dataloader = OmegaConf.create()


dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="odor_test", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


def get_odor_dict_test():
    return get_odor_dict('test')


def get_odor_meta(split):
    if split == 'train':
        pth = 'data/ODOR-v3/coco-style/annotations/instances_train2017.json'
        img_pth = 'data/ODOR-v3/coco-style/train2017'
    elif split == 'test':
        pth = 'data/ODOR-v3/coco-style/annotations/instances_val2017.json'
        img_pth = 'data/ODOR-v3/coco-style/val2017'
    else:
        raise Exception

    with open(pth) as f:
        coco = json.load(f)
    class_names = [cat['name'] for cat in coco['categories']]

    return {
        "json_file": pth,
        "image_root": img_pth,
        "class_names": class_names
    }


def get_odor_dict(split):
    meta = get_odor_meta(split)

    pth = meta['json_file']
    img_pth = meta['image_root']

    with open(pth) as f:
        coco_anns = json.load(f)

    records = []

    imid_to_anns = {}
    for ann in coco_anns['annotations']:
        if ann['image_id'] not in imid_to_anns.keys():
            imid_to_anns[ann['image_id']] = [ann]
        else:
            im_anns = imid_to_anns[ann['image_id']]
            im_anns.append(ann)
            imid_to_anns[ann['image_id']] = im_anns

    for img in coco_anns['images']:
        img['file_name'] = f'{img_pth}/{img["file_name"]}'
        im_anns = imid_to_anns[img['id']]
        for ann in im_anns:
            ann['bbox_mode'] = BoxMode.XYWH_ABS
            ann['segmentation'] = []
            del ann['image_id']
        img['annotations'] = im_anns
        img['image_id'] = img['id']
        records.append(img)

    return records


DatasetCatalog.register('odor_test', get_odor_dict_test)


for split in ['train', 'test']:
    ds_name = f'odor_{split}'
    meta = get_odor_meta(split)

    MetadataCatalog.get(ds_name).set(
        thing_classes=meta['class_names'], json_file=meta['json_file'], image_root=meta['image_root'], evaluator_tyoe='coco')
