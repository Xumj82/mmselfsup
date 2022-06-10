_base_ = [
    '../_base_/models/mocov3_vit-small-p16.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]


# dataset settings
data_source = 'DdsmBreast'
dataset_type = 'BreastDuoViewDataset'
max_pixel_val = 65535
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.2,
    #             hue=0.1)
    #     ],
    #     p=0.8),
    # dict(type='RandomGrayscale', p=0.2),
    # dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=1.),
    # dict(type='Solarization', p=0.),
    dict(type='RandomHorizontalFlip'),
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    # dict(
    #     type='RandomAppliedTrans',
    #     transforms=[
    #         dict(
    #             type='ColorJitter',
    #             brightness=0.4,
    #             contrast=0.4,
    #             saturation=0.2,
    #             hue=0.1)
    #     ],
    #     p=0.8),
    # dict(type='RandomGrayscale', p=0.2),
    # dict(type='GaussianBlur', sigma_min=0.1, sigma_max=2.0, p=0.1),
    # dict(type='Solarization', p=0.2),
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline1.extend(
        [dict(type='ToTensor'),
        #  dict(type='DuoViewImageToTensor'),
         dict(type='LinearNormalize', max_val=max_pixel_val),])
    train_pipeline2.extend(
        [dict(type='ToTensor'),
        #  dict(type='DuoViewImageToTensor'),
         dict(type='LinearNormalize', max_val=max_pixel_val),])

# dataset summary
data = dict(
    samples_per_gpu=1,  # 256*16(gpu)=4096
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='/home/xumingjie/Desktop/ddsm_breast/',
            ann_file='/home/xumingjie/Desktop/ddsm_breast/seq_lv_train_set.csv',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2],
        prefetch=prefetch,
    ))



# MoCo v3 use the same momentum update method as BYOL
custom_hooks = [dict(type='MomentumUpdateHook')]

# optimizer
optimizer = dict(type='AdamW', lr=2.4e-3, weight_decay=0.1)

# fp16
fp16 = dict(loss_scale='dynamic')

# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=10, max_keep_ckpts=3)