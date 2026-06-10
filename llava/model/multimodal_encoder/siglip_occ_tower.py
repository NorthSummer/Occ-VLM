# Copyright (c) OpenRobotLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch


from llava.model.multimodal_encoder.embodiedscan.registry import MODELS, TASK_UTILS
from llava.model.multimodal_encoder.embodiedscan.utils import ConfigType, OptConfigType


# ----------------- Config portion -----------------
_base_ = ['../default_runtime.py']
n_points = 100000

# origin for multi-view scannet is set to 0.5
# -1.28~1.28 -> -0.78~1.78
point_cloud_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78]

prior_generator = dict(type='AlignedAnchor3DRangeGenerator',
                       ranges=[[-3.2, -3.2, -1.28, 3.2, 3.2, 1.28]],
                       rotations=[.0])

dense_generator = dict(type='AlignedAnchor3DRangeGenerator',
                       ranges=[[-4.8, -4.8, -1.92, 3.2, 3.2, 1.92]],
                       rotations=[.0])

siglip_occ_tower_config = dict(
    type='DenseFusionOccPredictor_SigLip',
    use_valid_mask=False,
    use_xyz_feat=True,
    point_cloud_range=point_cloud_range,
    data_preprocessor=None,
    adapter=dict(type='SigLipBackboneAdapter',
                base_model_name="google/siglip-so400m-patch14-384", 
                copy_indices=[6, 12, 18], 
                out_channels=[256, 512, 1024, 1152],
                freeze_backbone=True),
    neck=dict(type='mmdet.FPN',
              in_channels=[256, 512, 1024, 1152],
              out_channels=256,
              num_outs=4),
    neck_3d=dict(type='IndoorImVoxelNeck',
                 in_channels=256,
                 out_channels=128,
                 n_blocks=[1, 1, 1]),
    bbox_head=dict(
        type='ImVoxelOccHead',
        volume_h=[20, 10, 5],
        volume_w=[20, 10, 5],
        volume_z=[8, 4, 2],
        num_classes=81,  # TO Be changed
        in_channels=[128, 128, 128],
        use_semantic=True),
    prior_generator=prior_generator,
    dense_generator=dense_generator,
    n_voxels=[40, 40, 16],  # voxel_size=(.16, .16, .16)
    n_dense_points=[240, 240, 96],
    coord_type='DEPTH',
    from_pretrained='/home/heliulu/ljn/EmbodiedScan/work_dirs/mv-occ-siglip/epoch_28.pth',
)



if __name__ == "__main__":
    """
    Build the model defined by the `model` dict above using the project's MODELS
    registry. This will instantiate the DenseFusionOccPredictor_SigLip and its
    submodules (neck, neck_3d, bbox_head, prior_generator, ...).

    Usage:
      python dense_fusion_occ_siglip.py

    Note: this script assumes it is run inside the project where
    `embodiedscan.registry.MODELS` (and other required registries) are importable
    and all referenced submodules (e.g., IndoorImVoxelNeck, ImVoxelOccHead,
    SigLipBackboneAdapter, SigLipImageProcessor, TASK_UTILS builders, etc.)
    are registered.
    """
    import traceback
    import torch

    try:
        print("Building model from local `model` config...")
        built_model = MODELS.build(siglip_occ_tower_config)
        print("Model built successfully.")
        # Move to cpu for safe introspection
        device = torch.device("cpu")
        try:
            built_model.to(device)
        except Exception:
            # Some modules (like certain wrappers) might not support .to()
            pass

        # Basic info
        print("\nModel summary (top-level):")
        print(built_model)

        total_params = sum(p.numel() for p in built_model.parameters())
        trainable_params = sum(p.numel() for p in built_model.parameters() if p.requires_grad)
        print(f"\nParameters: total={total_params:,}, trainable={trainable_params:,}")

        # Helpful attributes
        print("\nAttributes:")
        print(f"  with_neck: {getattr(built_model, 'with_neck', None)}")
        print(f"  with_neck_3d: {getattr(built_model, 'with_neck_3d', None)}")
        print(f"  coord_type: {getattr(built_model, 'coord_type', None)}")
        print(f"  n_voxels: {getattr(built_model, 'n_voxels', None)}")
        print(f"  point_cloud_range: {getattr(built_model, 'point_cloud_range', None)}")

        # Try a very small instantiation check where possible:
        # Many submodules expect actual data, so we only attempt to call .eval()
        try:
            built_model.eval()
            print("\nSet model to eval() successfully (basic sanity).")
        except Exception as e:
            print("\nCould not set model to eval():", e)

    except Exception as e:
        print("Failed to build model. Exception:")
        traceback.print_exc()
