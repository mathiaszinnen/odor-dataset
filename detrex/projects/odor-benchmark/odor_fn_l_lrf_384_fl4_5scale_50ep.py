from detrex.config import get_config

from .dino_focalnet_large_lrf_384_fl4_5scale_12ep import (
    train,
    optimizer,
    model,
)

EPOCH_ITERS = 4264

dataloader = get_config("common/data/odor.py").dataloader
model.num_classes = 139

# using 36ep scheduler
lr_multiplier = get_config("common/coco_schedule.py").lr_multiplier_36ep

# modify training config
train.max_iter = 50 * EPOCH_ITERS
train.init_checkpoint = "/net/cluster/zinnen/models/focalnet_large_lrf_384_fl4.pth"
train.output_dir = "./net/cluster/zinnen/detrex-output/odor3-tests"

train.eval_period = EPOCH_ITERS
train.checkpointer.period = EPOCH_ITERS

# using larger drop-path rate for longer training times
model.backbone.drop_path_rate = 0.4
