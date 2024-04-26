import json

from detectron2.engine import default_argument_parser, default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.structures.boxes import BoxMode
from tqdm import tqdm
import numpy as np

import torch


def to_coco(dt2_output):
    imgs = []
    annotations = []

    for img, preds in dt2_output:
        imgs.append({
            'id': img['id'],
            'file_name': img['file_name'],
            'height': img['height'],
            'width': img['width']
        })
        boxes = preds.get('pred_boxes')
        scores = preds.get('scores')
        classes = preds.get('pred_classes')

        for box, score, cls in zip(boxes, scores, classes):
            ann_id = len(annotations)
            box = box.cpu().numpy()
            box = BoxMode.convert(np.array([box]), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            annotations.append(
                {
                    'id': ann_id,
                    'img_id': img['id'],
                    'category_id': int(cls.cpu().item()),
                    'bbox': list(map(int, box[0])),
                    'score': float(score.cpu().item())
                }
            )

    return {
        'images': imgs,
        'annotations': annotations
    }


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    out_path = '/hdd/detectors/detrex/out.json'
    device = cfg.train.device
    model = instantiate(cfg.model)
    model.to(device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()

    dataloader = instantiate(cfg.dataloader.test)

    output = []
    for img in tqdm(dataloader):
        with torch.no_grad():
            model_preds = model(img)
        output.append((img[0],model_preds[0]['instances']))

    out = to_coco(output)
    with open(out_path, 'w') as f:
        json.dump(out, f)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    main(args)
