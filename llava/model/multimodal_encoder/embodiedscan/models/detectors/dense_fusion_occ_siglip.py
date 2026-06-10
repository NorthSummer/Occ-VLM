# Copyright (c) OpenRobotLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from mmengine.structures import InstanceData

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = None
    pass

from mmengine.model import BaseModel
from llava.utils import rank0_print

from llava.model.multimodal_encoder.embodiedscan.registry import MODELS, TASK_UTILS
from llava.model.multimodal_encoder.embodiedscan.structures.bbox_3d import get_proj_mat_by_coord_type
from llava.model.multimodal_encoder.embodiedscan.utils import ConfigType, OptConfigType
from llava.model.multimodal_encoder.embodiedscan.utils.typing_config import (ForwardResults, InstanceList,
                                              SampleList)

from llava.model.multimodal_encoder.embodiedscan.models.layers.fusion_layers.point_fusion import (batch_point_sample,
                                                 point_sample, batch_points_to_pixel_coords)

from llava.model.multimodal_encoder.embodiedscan.models.backbones import SigLipBackboneAdapter
from llava.model.multimodal_encoder.embodiedscan.models.backbones import SigLipImageProcessor

@MODELS.register_module()
class DenseFusionOccPredictor_SigLip(BaseModel):
    """Dense Fusion framework for occupancy prediction.

    Args:
        backbone (:obj:`ConfigDict` or dict): The image backbone config.
        backbone_3d (:obj:`ConfigDict` or dict): The 3D backbone config.
        neck (:obj:`ConfigDict` or dict): The image neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior grid generator
            config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        use_valid_mask (bool): Whether to use valid masks to handle
            visible voxels. Defaults to False.
        use_xyz_feat (bool): Whether to use xyz features.
            Defaults to False.
        point_cloud_range (list]): Point cloud range, [x_min, y_min, z_min,
            x_max, y_max, z_max], e.g., [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78].
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 adapter: ConfigType,
                 neck: ConfigType,
                 neck_3d: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 dense_generator: ConfigType,
                 n_voxels: List,
                 n_dense_points: List,
                 coord_type: str,
                 use_valid_mask=True,
                 use_xyz_feat: bool = False,
                 point_cloud_range=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None,
                 from_pretrained: str = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        
        self.adapter = MODELS.build(adapter)
        # self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        if neck_3d is not None:
            self.neck_3d = MODELS.build(neck_3d)

        # SigLipBackboneAdapter(base_model_name="google/siglip-so400m-patch14-384", 
        #                                      copy_indices=[6, 12, 18], 
        #                                      out_channels=[256, 512, 1024, 1152])
        
        self.processor = SigLipImageProcessor()

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.n_voxels = n_voxels
        self.n_dense_points = n_dense_points
        self.point_cloud_range = point_cloud_range
        prior_range = prior_generator['ranges'][0]

        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.dense_generator = TASK_UTILS.build(dense_generator)

        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_valid_mask = use_valid_mask
        self.use_xyz_feat = use_xyz_feat

        self.is_loaded = True
        if from_pretrained is not None:
            self.from_pretrained = from_pretrained


    @property
    def with_neck(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_3d(self):
        """Whether the detector has a 3D neck."""
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    def load_model(self, device_map=None):
        if self.is_loaded:
            rank0_print("{} is already loaded, `load_model` called again, skipping.")
            return

        if self.from_pretrained is not None:
            ckpt = torch.load(self.from_pretrained, map_location='cuda')
            state_dict = ckpt.get('state_dict', ckpt)
            self.load_state_dict(state_dict, strict=True)
            print("successfully loaded occ_tower ckpt")

        self.is_loaded = True

    def resample_feat(self, vfm_feat, prior_points, video_dict):

        # vfm_feat = F.interpolate(vfm_feat, size=(96, 96), mode='bilinear', align_corners=False)
        x = vfm_feat.unsqueeze(0)

        batch_img_metas= video_dict['data_meta_info']
        img_scale_factor = (prior_points.new_tensor(
                video_dict['scale_factor'][:2])
                                if 'scale_factor' in video_dict.keys() else 1)                        
        img_shape = video_dict['img_shape']

        if 'origin' in batch_img_metas[0]['depth2img'].keys():
            assert len(batch_img_metas) == 1, 'only support batch_size=1 here'
            prior_points += prior_points.new_tensor(
                batch_img_metas[0]['depth2img']['origin'])
            # For calibration with original ImVoxelNet implementation
            # prior_points += prior_points.new_tensor([-0.08, -0.08, -0.08])

        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):

            img_flip = False
            img_crop_offset =  0
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View ImVoxelNet
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []
                # Support different intrinsic matrices for different images
                # if the original intrinsic is only a matrix
                # we will simply copy it to construct the intrinsic matrix list
                # in MultiViewPipeline
                # assert isinstance(proj_mat['intrinsic'], list)
                for proj_idx in range(len(proj_mat['extrinsic'])):
                    intrinsic = vfm_feat.new_tensor(proj_mat['intrinsic'])
                    extrinsic = vfm_feat.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                proj_mat = torch.stack(projection).to(x.device)
                # feature torch.Size([32, 256, 96, 96])
                volume = batch_point_sample(
                    img_meta,
                    img_features=feature,
                    points=prior_points,
                    proj_mat=proj_mat,
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=(384, 384), #feature.shape[-2:], #img.shape[-2:],
                    img_shape=img_shape,
                    aligned=False)

            volumes.append(
                volume)

        sampled_vfm_feat = torch.stack(volumes)

        return sampled_vfm_feat

    def project_dense_points(self, video_dict, occ_pts):
        batch_img_metas = video_dict['data_meta_info']

        dense_points = occ_pts

        img_scale_factor = (dense_points.new_tensor(
                video_dict['scale_factor'][:2])
                                if 'scale_factor' in video_dict.keys() else 1)                        
        img_shape = video_dict['img_shape']

        if 'origin' in batch_img_metas[0]['depth2img'].keys():
            assert len(batch_img_metas) == 1, 'only support batch_size=1 here'

        assert len(batch_img_metas) == 1
        for img_meta in batch_img_metas:
            img_flip = False
            img_crop_offset =  0
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View ImVoxelNet
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []

                for proj_idx in range(len(proj_mat['extrinsic'])):
                    intrinsic = dense_points.new_tensor(proj_mat['intrinsic'])
                    extrinsic = dense_points.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                proj_mat = torch.stack(projection).to(dense_points.device)
                # feature torch.Size([32, 256, 96, 96])
                dense_points_2d, dense_2d_mask, depth_map = batch_points_to_pixel_coords(
                    img_meta,
                    points=dense_points,              # (B, N, 3) 或 (N, 3)
                    proj_mat=proj_mat,            # (B, 4, 4) 投影矩阵
                    coord_type=self.coord_type,             # 'LIDAR' / 'CAMERA' / 'DEPTH'
                    img_scale_factor=img_scale_factor,    # (w_scale, h_scale) 或 (B, 2)
                    img_crop_offset=img_crop_offset,     # (w_offset, h_offset) 或 (B, 2)
                    img_flip=img_flip,
                    img_pad_shape=(384, 384),   # (H, W) padding后
                    img_shape=img_shape)      # (H, W) padding前
        

                # 2. 保存
                save_depth_map(
                    depth=depth_map,
                    valid_mask=dense_2d_mask,
                    save_path="./output/depth",
                    filename="proj_depth",
                    format="npy",
                )

                # 3. 可视化
                show_depth_comparison(
                    depth_pred=depth_map[0].cpu().numpy(),
                    valid_mask=dense_2d_mask[0].cpu().numpy(),
                    clip_range=(0.5, 10.0),
                    save_path="./output/depth/vis.png",
)       

        return dense_points_2d, dense_2d_mask

    def extract_feat(self, img, video_dict):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
             - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
             - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """
        # 1. Extract the feature volume from images
        # img = batch_inputs_dict['imgs']
        # img = self.processor.preprocess(img, return_tensors='pt')['pixel_values']

        batch_img_metas = video_dict['data_meta_info']
        batch_size = 1
    
        siglip_feat, last_hidden = self.adapter(img.unsqueeze(0))

        siglip_feat[0] = torch.zeros_like(siglip_feat[0])
        siglip_feat[1] = torch.zeros_like(siglip_feat[1])
        siglip_feat[2] = torch.zeros_like(siglip_feat[2])

        x = self.neck(siglip_feat)[0]
        x = x.reshape([batch_size, -1] + list(x.shape)[1:])
        
        # x torch.Size([1, 32, 256, 32, 32])
        prior_points = self.prior_generator.grid_anchors(
            [self.n_voxels[::-1]], device=x.device)[0][:, :3]

        dense_points = self.dense_generator.grid_anchors(
            [self.n_dense_points[::-1]], device=x.device)[0][:, :3]
         
        img_scale_factor = (prior_points.new_tensor(
                video_dict['scale_factor'][:2])
                                if 'scale_factor' in video_dict.keys() else 1)                        
        img_shape = video_dict['img_shape']

        if 'origin' in batch_img_metas[0]['depth2img'].keys():
            assert len(batch_img_metas) == 1, 'only support batch_size=1 here'
            prior_points +=  prior_points.new_tensor(
                batch_img_metas[0]['depth2img']['origin'])
            dense_points +=  dense_points.new_tensor(
                batch_img_metas[0]['depth2img']['origin'])
            # For calibration with original ImVoxelNet implementation
            # prior_points += prior_points.new_tensor([-0.08, -0.08, -0.08])

        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):

            img_flip = False
            img_crop_offset =  0
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View ImVoxelNet
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []
                # Support different intrinsic matrices for different images
                # if the original intrinsic is only a matrix
                # we will simply copy it to construct the intrinsic matrix list
                # in MultiViewPipeline
                # assert isinstance(proj_mat['intrinsic'], list)
                for proj_idx in range(len(proj_mat['extrinsic'])):
                    intrinsic = img.new_tensor(proj_mat['intrinsic'])
                    extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                proj_mat = torch.stack(projection).to(x.device)
                # feature torch.Size([32, 256, 96, 96])

                volume = batch_point_sample(
                    img_meta,
                    img_features=feature,
                    points=prior_points,
                    proj_mat=proj_mat,
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img.shape[-2:], #feature.shape[-2:], #img.shape[-2:],
                    img_shape=img_shape,
                    aligned=False)

            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
        img_volume = torch.stack(volumes)
        x = img_volume
        x = self.neck_3d(x)

        return x, torch.stack(valid_preds).float(), last_hidden


    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                mode: str, # "predict" or "resample"
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        assert mode == "predict" or "resample"
        if mode == "predict":
            x, valid_preds, vfm_last_hidden = self.extract_feat(batch_inputs_dict,
                                                batch_data_samples)
            # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
            # mask of visible voxels.
            if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
                x += (valid_preds, )

            results_list = self.bbox_head.predict(x, **kwargs)
            # predictions = self.add_occupancy_to_data_sample(
            #     batch_data_samples, results_list)
            return results_list, vfm_last_hidden
        else:
            pass

    def add_occupancy_to_data_sample(self, data_samples: SampleList, pred):
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_occupancy = pred[i]
        return data_samples

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )
        results = self.bbox_head.forward(x)
        return results

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'predict',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of operation.

        Returns:
            The return type depends on ``mode``.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
        """
        return self.predict(inputs, data_samples, mode, **kwargs)


    def add_pred_to_datasample(
        self,
        data_samples: SampleList,
        data_instances_3d: Optional[InstanceList] = None,
        data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None),\
               'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples


import os
import numpy as np
import cv2

def save_depth_map(
    depth,
    valid_mask,
    save_path: str = "./depth_output",
    filename: str = "depth",
    format: str = "npy",  # 'npy' / 'png' / 'exr'
    clip_range: Tuple[float, float] = (0.0, 10.0),
    save_raw: bool = True,
    save_visual: bool = True,
) -> dict:
    """
    保存深度图到磁盘，支持多种格式。
    
    Args:
        depth: (B, H, W) 或 (H, W) 深度图，单位：米
        valid_mask: (B, H, W) 或 (H, W) 有效区域掩码
        save_path: 保存目录
        filename: 文件名前缀
        format: 保存格式 ('npy' 保留精度 / 'png' 便于查看 / 'exr' 专业格式)
        clip_range: 深度裁剪范围 (min_depth, max_depth)，用于 png 可视化
        save_raw: 是否保存原始深度数据
        save_visual: 是否保存可视化彩色图
    
    Returns:
        stats: 保存统计信息
    """
    # 转换为 numpy
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()
    
    # 处理 batch 维度
    if depth.ndim == 3:
        batch_list = [depth[i] for i in range(depth.shape[0])]
        mask_list = [valid_mask[i] for i in range(valid_mask.shape[0])] if valid_mask is not None else [None] * depth.shape[0]
    else:
        batch_list = [depth]
        mask_list = [valid_mask] if valid_mask is not None else [None]
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    stats = {
        'saved_files': [],
        'depth_min': [],
        'depth_max': [],
        'valid_ratio': [],
    }
    
    for idx, (depth_map, valid_mask_map) in enumerate(zip(batch_list, mask_list)):
        # 统计信息
        if valid_mask_map is not None:
            valid_depth = depth_map[valid_mask_map]
            depth_min = float(valid_depth.min()) if valid_depth.size > 0 else 0.0
            depth_max = float(valid_depth.max()) if valid_depth.size > 0 else 0.0
            valid_ratio = float(valid_mask_map.sum()) / valid_mask_map.size
        else:
            depth_min = float(depth_map.min())
            depth_max = float(depth_map.max())
            valid_ratio = 1.0
        
        stats['depth_min'].append(depth_min)
        stats['depth_max'].append(depth_max)
        stats['valid_ratio'].append(valid_ratio)
        
        # 文件名
        file_suffix = f"_{idx:03d}" if len(batch_list) > 1 else ""
        
        # ============ 保存原始深度数据 ============
        if save_raw:
            if format == "npy":
                raw_path = os.path.join(save_path, f"{filename}{file_suffix}.npy")
                np.save(raw_path, depth_map)
                stats['saved_files'].append(raw_path)
            
            elif format == "exr":
                raw_path = os.path.join(save_path, f"{filename}{file_suffix}.exr")
                # OpenCV 保存 EXR 需要 (H, W, C) 格式
                depth_exr = depth_map.astype(np.float32)
                cv2.imwrite(raw_path, depth_exr)
                stats['saved_files'].append(raw_path)
            
            elif format == "png":
                # PNG 需要归一化到 0-255，会损失精度
                raw_path = os.path.join(save_path, f"{filename}{file_suffix}.png")
                depth_norm = np.clip((depth_map - clip_range[0]) / (clip_range[1] - clip_range[0]), 0, 1)
                depth_uint16 = (depth_norm * 65535).astype(np.uint16)  # 用 uint16 保留更多精度
                cv2.imwrite(raw_path, depth_uint16)
                stats['saved_files'].append(raw_path)
        
        # ============ 保存可视化彩色图 ============
        if save_visual:
            vis_path = os.path.join(save_path, f"{filename}{file_suffix}_vis.png")
            vis_img = visualize_depth(depth_map, valid_mask_map, clip_range)
            cv2.imwrite(vis_path, vis_img)
            stats['saved_files'].append(vis_path)
        
        # ============ 保存掩码 (可选) ============
        if valid_mask_map is not None:
            mask_path = os.path.join(save_path, f"{filename}{file_suffix}_mask.png")
            cv2.imwrite(mask_path, (valid_mask_map * 255).astype(np.uint8))
            stats['saved_files'].append(mask_path)
    
    # 打印统计
    # print(f"[Save Depth] Saved {len(stats['saved_files'])} files to {save_path}")
    # print(f"  Depth Range: [{min(stats['depth_min']):.3f}, {max(stats['depth_max']):.3f}] m")
    # print(f"  Valid Ratio: {np.mean(stats['valid_ratio'])*100:.2f}%")
    
    return stats


def load_depth_map(
    load_path: str,
    filename: str = "depth",
    format: str = "npy",
    idx: int = 0,
    clip_range: Tuple[float, float] = (0.0, 10.0),
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    读取保存的深度图。
    
    Args:
        load_path: 加载目录
        filename: 文件名前缀
        format: 文件格式 ('npy' / 'png' / 'exr')
        idx: batch 索引
        clip_range: 深度范围 (用于 png 反归一化)
    
    Returns:
        depth: (H, W) 深度图
        valid_mask: (H, W) 有效掩码 (如果存在)
    """
    file_suffix = f"_{idx:03d}" if idx > 0 else ""
    
    # 读取深度
    if format == "npy":
        depth_path = os.path.join(load_path, f"{filename}{file_suffix}.npy")
        depth = np.load(depth_path)
    
    elif format == "exr":
        depth_path = os.path.join(load_path, f"{filename}{file_suffix}.exr")
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    
    elif format == "png":
        depth_path = os.path.join(load_path, f"{filename}{file_suffix}.png")
        depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        # 反归一化
        depth = depth_uint16 / 65535.0 * (clip_range[1] - clip_range[0]) + clip_range[0]
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # 读取掩码 (如果存在)
    mask_path = os.path.join(load_path, f"{filename}{file_suffix}_mask.png")
    valid_mask = None
    if os.path.exists(mask_path):
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 128
    
    return depth, valid_mask

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def visualize_depth(
    depth,
    valid_mask,
    clip_range = (0.0, 10.0),
    colormap: str = "viridis",
    show_invalid: bool = True,
) -> np.ndarray:
    """
    将深度图转换为彩色可视化图像。
    
    Args:
        depth: (H, W) 深度图
        valid_mask: (H, W) 有效区域掩码
        clip_range: 深度显示范围 (min, max)
        colormap: matplotlib colormap 名称
        show_invalid: 是否用特殊颜色显示无效区域
    
    Returns:
        vis_img: (H, W, 3) RGB 图像 (0-255, uint8)
    """
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if valid_mask is not None and isinstance(valid_mask, torch.Tensor):
        valid_mask = valid_mask.detach().cpu().numpy()
    
    # 裁剪深度范围
    depth_vis = np.clip(depth, clip_range[0], clip_range[1])
    
    # 归一化到 0-1
    depth_norm = (depth_vis - clip_range[0]) / (clip_range[1] - clip_range[0] + 1e-8)
    
    # 应用 colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(depth_norm)[..., :3]  # (H, W, 3)
    
    # 处理无效区域
    if valid_mask is not None and show_invalid:
        # 无效区域显示为黑色
        colored[~valid_mask] = [0, 0, 0]
    
    # 转换为 uint8
    vis_img = (colored * 255).astype(np.uint8)
    
    return vis_img


def show_depth_comparison(
    depth_pred: np.ndarray,
    depth_gt: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    clip_range: Tuple[float, float] = (0.0, 10.0),
    save_path: Optional[str] = None,
) -> None:
    """
    并排显示预测深度和 GT 深度的对比图。
    
    Args:
        depth_pred: 预测深度图 (H, W)
        depth_gt: Ground Truth 深度图 (H, W)
        valid_mask: 有效区域掩码
        clip_range: 深度显示范围
        save_path: 保存路径 (如果为 None 则直接显示)
    """
    fig, axes = plt.subplots(1, 3 if depth_gt is not None else 1, figsize=(15, 5))
    if depth_gt is None:
        axes = [axes]
    
    # 预测深度
    vis_pred = visualize_depth(depth_pred, valid_mask, clip_range)
    axes[0].imshow(vis_pred)
    axes[0].set_title(f"Predicted Depth\n[{clip_range[0]}-{clip_range[1]}m]")
    axes[0].axis('off')
    
    # GT 深度 (如果有)
    if depth_gt is not None:
        vis_gt = visualize_depth(depth_gt, valid_mask, clip_range)
        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth Depth\n[{clip_range[0]}-{clip_range[1]}m]")
        axes[1].axis('off')
        
        # 误差图
        if valid_mask is not None:
            error = np.abs(depth_pred - depth_gt) * valid_mask
            error_norm = np.clip(error / clip_range[1], 0, 1)
        else:
            error = np.abs(depth_pred - depth_gt)
            error_norm = np.clip(error / clip_range[1], 0, 1)
        
        cmap_error = plt.get_cmap("hot")
        vis_error = (cmap_error(error_norm)[..., :3] * 255).astype(np.uint8)
        if valid_mask is not None:
            vis_error[~valid_mask] = [0, 0, 0]
        
        axes[2].imshow(vis_error)
        axes[2].set_title(f"Absolute Error\nMean: {error[valid_mask].mean() if valid_mask is not None else error.mean():.3f}m")
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print(f"[Visual] Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()