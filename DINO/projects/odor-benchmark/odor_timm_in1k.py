from detectron2.config import LazyCall as L
from detectron2.modeling import ShapeSpec
from detrex.config import get_config

# inherit configs from "dino_r50_4scale_12ep"
from .dino_r50_4scale_12ep import (
    train,
    dataloader,
    optimizer,
    lr_multiplier,
)
from .models.dino_r50 import model


EPOCH_ITERS = 4264

dataloader = get_config('common/data/odor.py').dataloader

from detrex.modeling.backbone import TimmBackbone

# modify backbone configs
model.backbone = L(TimmBackbone)(
    model_name="resnet50_miil_1k",
    pretrained=True,
    # specify the return nodes
    in_channels=3,
    out_indices=(1,2,3),
    norm_layer=FrozenBatchNorm2d,
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
train.output_dir = "./output/timm_in1k"
train.max_iter = 50 * EPOCH_ITERS
train.eval_period = EPOCH_ITERS
train.checkpointer.period = EPOCH_ITERS

dataloader.train.total_batch_size=8
dataloader.train.num_workers=2

lr_multiplier = get_config("common/coco_schedule.py").odor_multiplier_50ep
