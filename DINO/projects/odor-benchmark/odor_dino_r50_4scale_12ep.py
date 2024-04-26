from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from detrex.config import get_config

# inherit configs from "dino_r50_4scale_12ep"
from .dino_r50_4scale_12ep import (
    train,
    optimizer,
    lr_multiplier,
)
from .models.dino_r50 import model


EPOCH_ITERS = 4264

dataloader = get_config('common/data/odor.py').dataloader

from detrex.modeling.backbone import TorchvisionBackbone

# modify backbone configs
model.backbone = L(TorchvisionBackbone)(
    model_name="resnet50",
    pretrained=True,
    # specify the return nodes
    return_nodes = {
        "layer2": "res3",
        "layer3": "res4",
        "layer4": "res5",
    },
)

# modify neck configs
model.neck.input_shapes = {
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res3", "res4", "res5"]

model.num_classes = 139

# modify training configs
# train.init_checkpoint = ""
train.output_dir = "./output/rn50"
train.max_iter = 50 * EPOCH_ITERS
train.eval_period = EPOCH_ITERS
train.checkpointer.period = EPOCH_ITERS

dataloader.train.total_batch_size=8
dataloader.train.num_workers=2

lr_multiplier = get_config("common/coco_schedule.py").odor_multiplier_50ep
