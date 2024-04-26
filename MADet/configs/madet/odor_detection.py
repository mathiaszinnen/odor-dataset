# dataset settings
dataset_type = 'CocoDataset'
data_root = '../data/'


classes = ("ant", "camel", "jewellery", "frog", "physalis", "celery", "cauliflower", "pepper", "ranunculus", "chess flower", "cigarette", "matthiola", "cabbage", "earring", "dandelion", "neroli", "dragonfly", "hyacinth", "reptile/amphibia", "apricot", "snake", "lizard", "asparagus", "spring onion", "snowflake", "moth", "poppy", "columbine", "rabbit", "geranium", "crab", "radish", "big cat", "jan steen jug", "monkey", "snail", "bellflower", "lilac", "pot", "peony", "coffeepot", "hazelnut", "censer", "artichoke", "dahlia", "sniffing", "fly", "deer", "caterpillar", "garlic", "blackberry", "chalice", "lobster", "necklace", "bug", "insect", "prawn", "bracelet", "carrot", "cornflower", "pumpkin", "orange", "walnut", "cat", "daisy", "forget-me-not", "carafe", "match", "beer stein", "tobacco-box", "violet", "pomander", "bottle", "candle", "heliotrope", "wine bottle", "strawberry", "pomegranate", "whale", "lily of the valley", "iris", "tobacco", "olive", "tobacco-packaging", "meat", "daffodil", "melon", "fire", "petunia", "mushroom", "teapot", "ring", "pig", "ashtray", "cheese", "onion", "cup", "nut", "fig", "drinking vessel", "donkey", "holding the nose", "lily", "smoke", "bread", "currant", "glass without stem", "anemone", "mammal", "chimney", "smoking equipment", "bivalve", "butterfly", "gloves", "lemon", "horse", "plum", "jasmine", "pear", "glass with stem", "vegetable", "carnation", "jug", "goat", "fish", "apple", "tulip", "cherry", "cow", "animal corpse", "dog", "fruit", "bird", "rose", "peach", "sheep", "pipe", "grapes", "flower")


img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations_train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations_train.json',
        img_prefix=data_root + 'images',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations_test.json',
        img_prefix=data_root + 'images',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox'])
