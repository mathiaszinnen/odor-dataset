from detrex.config import get_config

from .dino_r50_4scale_12ep import (
    train,
    optimizer,
    lr_multiplier,
    model,
)

from detectron2.layers import ShapeSpec

EPOCH_ITERS = 4264

dataloader = get_config("common/data/odor.py").dataloader
model.num_classes = 139

# modify model config to generate 4 scale backbone features
# and 5 scale input features
model.backbone.out_features = ["res2", "res3", "res4", "res5"]

model.neck.input_shapes = {
    "res2": ShapeSpec(channels=256),
    "res3": ShapeSpec(channels=512),
    "res4": ShapeSpec(channels=1024),
    "res5": ShapeSpec(channels=2048),
}
model.neck.in_features = ["res2", "res3", "res4", "res5"]
model.neck.num_outs = 5
model.transformer.num_feature_levels = 5

# modify training config
train.output_dir = "./output/odor3_dino_r50_5scale_12ep"

# modify training config
train.max_iter = 50 * EPOCH_ITERS
# train.init_checkpoint = "/net/cluster/zinnen/models/focalnet_large_lrf_384_fl4.pth"
# train.output_dir = "./net/cluster/zinnen/detrex-output/odor3-tests"

train.eval_period = EPOCH_ITERS
train.checkpointer.period = EPOCH_ITERS
