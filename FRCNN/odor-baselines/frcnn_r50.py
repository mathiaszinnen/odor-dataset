_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/odor_instance.py',
    './schedule_50e.py', '../../configs/_base_/default_runtime.py'
]
