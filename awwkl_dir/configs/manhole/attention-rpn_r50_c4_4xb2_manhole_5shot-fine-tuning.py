_base_ = [
    '../../../configs/detection/attention_rpn/coco/attention-rpn_r50_c4_4xb2_coco_10shot-fine-tuning.py',
]

num_support_ways = 1
num_support_shots = 4
data_root = 'awwkl_dir/data/manhole_5-shot/'
classes=['Manhole']
data = dict(
    samples_per_gpu=1,
    train=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        dataset=dict(
            type='FewShotCocoDataset',
            ann_cfg=[dict(type='ann_file', ann_file=data_root+'train/annotation.json', method='Attention_RPN', setting='5-SHOT')],
            num_novel_shots=None,
            num_base_shots=None,
            classes=classes,
            img_prefix=data_root + 'train'),
        ),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'validate/annotation.json')],
        classes=classes,
        img_prefix=data_root + 'validate'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'test/annotation.json')],
        classes=classes,
        img_prefix=data_root + 'test'),
    model_init=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'train/annotation.json')],
        classes=classes,
        img_prefix=data_root + 'train',
    ))

model = dict(
    rpn_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots),
    roi_head=dict(
        num_support_ways=num_support_ways,
        num_support_shots=num_support_shots,
        bbox_head=dict(num_classes=1)))

runner = dict(max_iters=1200)
evaluation = dict(interval=50)
checkpoint_config = dict(interval=50)

# Use pre-trained model for better performance
load_from = ('work_dirs/downloaded/attention-rpn_r50_c4_4xb2_coco_base-training_20211102_003348-da28cdfd.pth')