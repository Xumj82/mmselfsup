_base_ = [
    # '../_base_/models/mocov3_vit-small-p16.py',
    '../_base_/schedules/adamw_coslr-300e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MoCoV3',
    base_momentum=0.99,
    # backbone=dict(
    #     type='VisionTransformer',
    #     arch='mocov3-small',  # embed_dim = 384
    #     img_size=(1120,896),
    #     in_channels=1,
    #     patch_size=16,
    #     stop_grad_conv1=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=1,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeck',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=3,
        with_bias=False,
        with_last_bn=True,
        with_last_bn_affine=False,
        with_last_bias=False,
        with_avg_pool=True,
        vit_backbone=False,
        # norm_cfg=dict(type='LN'),
        ),
    head=dict(
        type='MoCoV3Head',
        predictor=dict(
            type='NonLinearNeck',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            num_layers=2,
            with_bias=False,
            with_last_bn=True,
            with_last_bn_affine=False,
            with_last_bias=False,
            with_avg_pool=False),
        temperature=0.2))

# dataset settings
data_source = 'DdsmBreast'
dataset_type = 'BreastDuoViewDataset'
max_pixel_val = 65535
img_norm_cfg = None
train_pipeline1 = [
    # dict(type='RandRotate',range=0.2, prob=1.0),
    dict(type='RandAffine',rotate_range=0.2,shear_range=0.2,translate_range=0.2, scale_range=0.2,spatial_size=None,prob=1.0),
    dict(type='Rand2DElastic',
         spacing = (30, 30),
         magnitude_range = [5, 6],
         prob = 1.0,)
]
train_pipeline2 = [
    # dict(type='RandRotate',range=0.2, prob=1.0),
    dict(type='RandAffine',rotate_range=0.2,shear_range=0.2,translate_range=0.2, scale_range=0.2,spatial_size=None,prob=1.0),
    dict(type='Rand2DElastic',
         spacing = (30, 30),
         magnitude_range = [5, 6],
         prob = 1.0,)
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline1.extend(
        [
        dict(type='MinMaxNormalize'),
        dict(type='ToTensor'),
        #  dict(type='DuoViewImageToTensor'),
         ])
    train_pipeline2.extend(
       
        [
            dict(type='MinMaxNormalize'),
            dict(type='ToTensor'),
        #  dict(type='DuoViewImageToTensor'),
         ])

# dataset summary
data = dict(
    samples_per_gpu=1,  # 256*16(gpu)=4096
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            img_shape=(1,1120,896),
            data_prefix='/home/xumingjie/Desktop/ddsm_breast/ddsm_breast/',
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