_base_ = [
    '../../configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning.py',
]

data_root = 'awwkl_dir/data/manhole_10-shot/'
data = dict(
    samples_per_gpu=1,
    train=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'train/annotation.json', method='FSCE', setting='10SHOT')],
        num_novel_shots=None,
        num_base_shots=None,
        classes=['Manhole'],
        img_prefix=data_root + 'train'),
    val=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'validate/annotation.json')],
        classes=['Manhole'],
        img_prefix=data_root + 'validate'),
    test=dict(
        type='FewShotCocoDataset',
        ann_cfg=[dict(type='ann_file', ann_file=data_root+'test/annotation.json')],
        classes=['Manhole'],
        img_prefix=data_root + 'test'),
    )

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

evaluation = dict(interval=10)
checkpoint_config = dict(interval=10)

# Use pre-trained model for better performance
load_from = ('work_dirs/downloaded/tfa_r101_fpn_coco_base-training.pth')
