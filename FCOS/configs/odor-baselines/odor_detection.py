_base_ = '../_base_/datasets/coco_detection.py'

data_root = '/hdd/datasets/odor-dataset/data'

classes = ("ant", "camel", "jewellery", "frog", "physalis", "celery", "cauliflower", "pepper", "ranunculus", "chess flower", "cigarette", "matthiola", "cabbage", "earring", "dandelion", "neroli", "dragonfly", "hyacinth", "reptile/amphibia", "apricot", "snake", "lizard", "asparagus", "spring onion", "snowflake", "moth", "poppy", "columbine", "rabbit", "geranium", "crab", "radish", "big cat", "jan steen jug", "monkey", "snail", "bellflower", "lilac", "pot", "peony", "coffeepot", "hazelnut", "censer", "artichoke", "dahlia", "sniffing", "fly", "deer", "caterpillar", "garlic", "blackberry", "chalice", "lobster", "necklace", "bug", "insect", "prawn", "bracelet", "carrot", "cornflower", "pumpkin", "orange", "walnut", "cat", "daisy", "forget-me-not", "carafe", "match", "beer stein", "tobacco-box", "violet", "pomander", "bottle", "candle", "heliotrope", "wine bottle", "strawberry", "pomegranate", "whale", "lily of the valley", "iris", "tobacco", "olive", "tobacco-packaging", "meat", "daffodil", "melon", "fire", "petunia", "mushroom", "teapot", "ring", "pig", "ashtray", "cheese", "onion", "cup", "nut", "fig", "drinking vessel", "donkey", "holding the nose", "lily", "smoke", "bread", "currant", "glass without stem", "anemone", "mammal", "chimney", "smoking equipment", "bivalve", "butterfly", "gloves", "lemon", "horse", "plum", "jasmine", "pear", "glass with stem", "vegetable", "carnation", "jug", "goat", "fish", "apple", "tulip", "cherry", "cow", "animal corpse", "dog", "fruit", "bird", "rose", "peach", "sheep", "pipe", "grapes", "flower")
metainfo=dict(
    classes=classes
)

train_dataloader=dict(
    dataset=dict(
        data_root=data_root,
        ann_file='instances_train.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo
    )
)
val_dataloader=dict(
    dataset=dict(
        data_root=data_root,
        ann_file='instances_test.json',
        data_prefix=dict(img='images/'),
        metainfo=metainfo
    )
)
test_dataloader=val_dataloader

val_evaluator=dict(
    ann_file=data_root+'/instances_test.json'
)
test_evaluator=val_evaluator