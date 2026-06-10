# Copyright (c) OpenMMLab and OpenRobotLab. All rights reserved.
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from llava.model.multimodal_encoder.embodiedscan.registry import MODELS
from llava.model.multimodal_encoder.embodiedscan.structures.bbox_3d import (batch_points_cam2img,
                                             get_proj_mat_by_coord_type,
                                             points_cam2img, points_img2cam)
from llava.model.multimodal_encoder.embodiedscan.structures.points import get_points_type
from llava.model.multimodal_encoder.embodiedscan.utils import ConfigType


def apply_3d_transformation(pcd: Tensor,
                            coord_type: str,
                            img_meta: dict,
                            reverse: bool = False) -> Tensor:
    """Apply transformation to input point cloud.

    Args:
        pcd (Tensor): The point cloud to be transformed.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_meta(dict): Meta info regarding data transformation.
        reverse (bool): Reversed transformation or not. Defaults to False.

    Note:
        The elements in img_meta['transformation_3d_flow']:

            - "T" stands for translation;
            - "S" stands for scale;
            - "R" stands for rotation;
            - "HF" stands for horizontal flip;
            - "VF" stands for vertical flip.

    Returns:
        Tensor: The transformed point cloud.
    """

    dtype = pcd.dtype
    device = pcd.device

    pcd_rotate_mat = (torch.tensor(img_meta['pcd_rotation'],
                                   dtype=dtype,
                                   device=device) if 'pcd_rotation' in img_meta
                      else torch.eye(3, dtype=dtype, device=device))

    pcd_scale_factor = (img_meta['pcd_scale_factor']
                        if 'pcd_scale_factor' in img_meta else 1.)

    pcd_trans_factor = (torch.tensor(
        img_meta['pcd_trans'], dtype=dtype, device=device)
                        if 'pcd_trans' in img_meta else torch.zeros(
                            (3), dtype=dtype, device=device))

    pcd_horizontal_flip = img_meta[
        'pcd_horizontal_flip'] if 'pcd_horizontal_flip' in \
        img_meta else False

    pcd_vertical_flip = img_meta[
        'pcd_vertical_flip'] if 'pcd_vertical_flip' in \
        img_meta else False

    flow = img_meta['transformation_3d_flow'] \
        if 'transformation_3d_flow' in img_meta else []

    pcd = pcd.clone()  # prevent inplace modification
    pcd = get_points_type(coord_type)(pcd)

    horizontal_flip_func = partial(pcd.flip, bev_direction='horizontal') \
        if pcd_horizontal_flip else lambda: None
    vertical_flip_func = partial(pcd.flip, bev_direction='vertical') \
        if pcd_vertical_flip else lambda: None
    if reverse:
        scale_func = partial(pcd.scale, scale_factor=1.0 / pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=-pcd_trans_factor)
        # pcd_rotate_mat @ pcd_rotate_mat.inverse() is not
        # exactly an identity matrix
        # use angle to create the inverse rot matrix neither.
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat.inverse())

        # reverse the pipeline
        flow = flow[::-1]
    else:
        scale_func = partial(pcd.scale, scale_factor=pcd_scale_factor)
        translate_func = partial(pcd.translate, trans_vector=pcd_trans_factor)
        rotate_func = partial(pcd.rotate, rotation=pcd_rotate_mat)

    flow_mapping = {
        'T': translate_func,
        'S': scale_func,
        'R': rotate_func,
        'HF': horizontal_flip_func,
        'VF': vertical_flip_func
    }
    for op in flow:
        assert op in flow_mapping, f'This 3D data '\
            f'transformation op ({op}) is not supported'
        func = flow_mapping[op]
        func()

    return pcd.coord


def point_sample(img_meta: dict,
                 img_features: Tensor,
                 points: Tensor,
                 proj_mat: Tensor,
                 coord_type: str,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True,
                 valid_flag: bool = False) -> Tensor:
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): 1 x C x H x W image features.
        points (Tensor): Nx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): 4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.
        valid_flag (bool): Whether to filter out the points that outside
            the image and with depth smaller than 0. Defaults to False.

    Returns:
        Tensor: NxC image features sampled by point coordinates.
    """

    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)

    # project points to image coordinate
    if valid_flag:
        proj_pts = points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
    img_coors -= img_crop_offset

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=1).unsqueeze(0).unsqueeze(0)  # Nx2 -> 1x1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCx1xN feats

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze() < w) & (coor_x.squeeze() > 0) & (
            coor_y.squeeze() < h) & (coor_y.squeeze() > 0) & (depths > 0)
        valid_features = point_features.squeeze().t()
        valid_features[~valid] = 0
        return valid_features, valid  # (N, C), (N,)

    return point_features.squeeze().t()


def batch_points_to_pixel_coords(
    img_meta: dict,
    points: Tensor,              # (B, N, 3) 或 (N, 3)
    proj_mat: Tensor,            # (B, 4, 4) 投影矩阵
    coord_type: str,             # 'LIDAR' / 'CAMERA' / 'DEPTH'
    img_scale_factor: Tensor,    # (w_scale, h_scale) 或 (B, 2)
    img_crop_offset: Tensor,     # (w_offset, h_offset) 或 (B, 2)
    img_flip: bool,
    img_pad_shape: Tuple[int],   # (H, W) padding 后
    img_shape: Tuple[int],       # (H, W) padding 前
    img_output_shape: Tuple[int] = (384, 384),
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    批量将 3D 点云投影到图像，构建每个像素对应的 3D 坐标映射表和深度图。
    
    Returns:
        pixel_coords: (B, H, W, 3) 每个样本每个像素对应的 3D 坐标
        valid_mask:   (B, H, W)    True 表示该像素有有效的 3D 点投影
        depth_map:    (B, H, W)    深度图 (Camera Frame Z)
    """

    H, W = img_output_shape
    B = proj_mat.shape[0]
    
    # ✅ 修复 1: 统一使用 output_shape 进行所有判断和初始化
    h_out, w_out = H, W
    
    # Step 1: 坐标变换
    points_transformed = apply_3d_transformation(
        points, coord_type, img_meta, reverse=True
    )  # (N, 3) 或 (B, N, 3)
    
    # ✅ 修复 3: 正确处理 Batch 维度
    if points_transformed.ndim == 2:
        points_transformed = points_transformed.unsqueeze(0).repeat(B, 1, 1)
    elif points_transformed.ndim == 3 and points_transformed.shape[0] != B:
        points_transformed = points_transformed.repeat(B, 1, 1)
    # 如果已经是 (B, N, 3) 且 B 匹配，则不需要操作

    N = points_transformed.shape[1]

    # Step 2: 投影
    proj_pts = batch_points_cam2img(points_transformed, proj_mat, with_depth=True)
    pts_2d  = proj_pts[..., :2]   # (B, N, 2)
    depths  = proj_pts[..., 2]    # (B, N)

    # Step 3: scale + crop + flip
    img_coors  = pts_2d * img_scale_factor
    img_coors -= img_crop_offset
    coor_x = img_coors[..., 0]
    coor_y = img_coors[..., 1]

    # ✅ 修复 2: 翻转使用变换后的宽度
    if img_flip:
        # 使用 pad_shape 或 output_shape 作为翻转参考（变换后的宽度）
        coor_x = w_out - coor_x - 1  # -1 因为索引从 0 开始

    # Step 4: valid 判断（统一用 output_shape）
    valid = (
        (coor_x >= 0) & (coor_x < w_out) &
        (coor_y >= 0) & (coor_y < h_out) &
        (depths > 0)
    )  # (B, N)

    # Step 5: 构建映射表
    pixel_coords = torch.zeros(B, H, W, 3, dtype=points.dtype, device=points.device)
    valid_mask   = torch.zeros(B, H, W, dtype=torch.bool, device=points.device)
    depth_map    = torch.zeros(B, H, W, dtype=points.dtype, device=points.device)

    batch_idx = torch.arange(B, device=points.device)[:, None].expand(B, N)

    xi = coor_x.long().clamp(0, W - 1)
    yi = coor_y.long().clamp(0, H - 1)

    valid_flat  = valid.reshape(-1)
    batch_flat  = batch_idx.reshape(-1)
    xi_flat     = xi.reshape(-1)
    yi_flat     = yi.reshape(-1)
    depths_flat = depths.reshape(-1)
    points_flat = points_transformed.reshape(-1, 3)

    valid_indices = valid_flat.nonzero(as_tuple=True)[0]
    if len(valid_indices) > 0:
        b_valid = batch_flat[valid_indices]
        x_valid = xi_flat[valid_indices]
        y_valid = yi_flat[valid_indices]
        d_valid = depths_flat[valid_indices]
        p_valid = points_flat[valid_indices]

        # Step 6: 近点优先（从远到近排序，近点最后写入覆盖远点）
        sort_idx = torch.argsort(d_valid, descending=True)
        b_valid  = b_valid[sort_idx]
        x_valid  = x_valid[sort_idx]
        y_valid  = y_valid[sort_idx]
        d_valid  = d_valid[sort_idx]
        p_valid  = p_valid[sort_idx]

        pixel_coords[b_valid, y_valid, x_valid] = p_valid
        valid_mask[b_valid, y_valid, x_valid]   = True
        depth_map[b_valid, y_valid, x_valid]    = d_valid

    # ✅ 修复 1: 统计使用 output_shape
    valid_pixels = valid_mask.sum().item()
    total_pixels = B * H * W
    print(f"[Pixel Stats] {valid_pixels}/{total_pixels} pixels "
          f"({valid_pixels/total_pixels*100:.2f}%) have 3D projections")

    return pixel_coords, valid_mask, depth_map



def batch_point_sample(img_meta: dict,
                       img_features: Tensor,
                       points: Tensor,
                       proj_mat: Tensor,
                       coord_type: str,
                       img_scale_factor: Tensor,
                       img_crop_offset: Tensor,
                       img_flip: bool,
                       img_pad_shape: Tuple[int],
                       img_shape: Tuple[int],
                       aligned: bool = True,
                       padding_mode: str = 'zeros',
                       align_corners: bool = True,
                       valid_flag: bool = True) -> Tensor:
    """Batch version of point_sample.

    Args:
        img_meta (dict): Meta info.
        img_features (Tensor): B x C x H x W image features.
        points (Tensor): BxNx3 point cloud in LiDAR coordinates.
        proj_mat (Tensor): Bx4x4 transformation matrix.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.
        valid_flag (bool): Whether to filter out the points that outside
            the image and with depth smaller than 0. Defaults to False.

    Returns:
        Tensor: NxC image features sampled by point coordinates.
    """
    # apply transformation based on info in img_meta
    points = apply_3d_transformation(points,
                                     coord_type,
                                     img_meta,
                                     reverse=True)

    points = points.repeat(proj_mat.shape[0], 1, 1)

    # points range aligns with pc_range, 3.12 ..

    # project points to image coordinate
    if valid_flag: # True
        proj_pts = batch_points_cam2img(points, proj_mat, with_depth=True)
        pts_2d = proj_pts[..., :2]
        depths = proj_pts[..., 2]
    else:
        pts_2d = points_cam2img(points, proj_mat)

    # img transformation: scale -> crop -> flip
    # the image is resized by img_scale_factor
    img_coors = pts_2d[..., 0:2] * img_scale_factor  # BxNx2
    img_coors -= img_crop_offset # 0

    # grid sample, the valid grid range should be in [-1,1]
    coor_x, coor_y = torch.split(img_coors, 1, dim=2)  # each is BxNx1

    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        coor_x = ori_w - coor_x

    h, w = img_pad_shape
    norm_coor_y = coor_y / h * 2 - 1
    norm_coor_x = coor_x / w * 2 - 1
    grid = torch.cat([norm_coor_x, norm_coor_y],
                     dim=2).unsqueeze(1)  # BxNx2 -> Bx1xNx2

    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    point_features = F.grid_sample(
        img_features.to(torch.float16),
        grid.to(torch.float16), #.to(img_features.dtype),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners).to(img_features.dtype)   # BxCx1xN feats

    if valid_flag:
        # (N, )
        valid = (coor_x.squeeze(2) < w) & (coor_x.squeeze(2) > 0) & (
            coor_y.squeeze(2) < h) & (coor_y.squeeze(2) > 0) & (depths > 0)
        valid_num = valid.sum(dim=0)  # N,
        valid_features = point_features.squeeze(2).sum(dim=0).t()  # NxC
        valid = valid_num > 0
        if len(valid) != len(valid_features):
            print('valid shape:', valid.shape)
            print('features shape:', valid_features.shape)
            print('img meta:', img_meta)
        valid_features[~valid, :] = 0.
        valid_features /= torch.clamp(valid_num[:, None], min=1)
        return valid_features  # (N, C), (N,)

    return point_features.squeeze().sum(dim=0).t()  # (N,C)


@MODELS.register_module()
class PointFusion(BaseModule):
    """Fuse image features from multi-scale features.

    Args:
        img_channels (List[int] or int): Channels of image features.
            It could be a list if the input is multi-scale image features.
        pts_channels (int): Channels of point features
        mid_channels (int): Channels of middle layers
        out_channels (int): Channels of output fused features
        img_levels (List[int] or int): Number of image levels. Defaults to 3.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'. Defaults to 'LIDAR'.
        conv_cfg (:obj:`ConfigDict` or dict): Config dict for convolution
            layers of middle layers. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict): Config dict for normalization
            layers of middle layers. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict): Config dict for activation layer.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`Contigdict` or dict],
            optional): Initialization config dict. Defaults to None.
        activate_out (bool): Whether to apply relu activation to output
            features. Defaults to True.
        fuse_out (bool): Whether to apply conv layer to the fused features.
            Defaults to False.
        dropout_ratio (int or float): Dropout ratio of image features to
            prevent overfitting. Defaults to 0.
        aligned (bool): Whether to apply aligned feature fusion.
            Defaults to True.
        align_corners (bool): Whether to align corner when sampling features
            according to points. Defaults to True.
        padding_mode (str): Mode used to pad the features of points that do not
            have corresponding image features. Defaults to 'zeros'.
        lateral_conv (bool): Whether to apply lateral convs to image features.
            Defaults to True.
    """

    def __init__(self,
                 img_channels: Union[List[int], int],
                 pts_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 img_levels: Union[List[int], int] = 3,
                 coord_type: str = 'LIDAR',
                 conv_cfg: Optional[ConfigType] = None,
                 norm_cfg: Optional[ConfigType] = None,
                 act_cfg: Optional[ConfigType] = None,
                 init_cfg: Optional[Union[ConfigType,
                                          List[ConfigType]]] = None,
                 activate_out: bool = True,
                 fuse_out: bool = False,
                 dropout_ratio: Union[int, float] = 0,
                 aligned: bool = True,
                 align_corners: bool = True,
                 padding_mode: str = 'zeros',
                 lateral_conv: bool = True) -> None:
        super(PointFusion, self).__init__(init_cfg=init_cfg)
        if isinstance(img_levels, int):
            img_levels = [img_levels]
        if isinstance(img_channels, int):
            img_channels = [img_channels] * len(img_levels)
        assert isinstance(img_levels, list)
        assert isinstance(img_channels, list)
        assert len(img_channels) == len(img_levels)

        self.img_levels = img_levels
        self.coord_type = coord_type
        self.act_cfg = act_cfg
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        self.dropout_ratio = dropout_ratio
        self.img_channels = img_channels
        self.aligned = aligned
        self.align_corners = align_corners
        self.padding_mode = padding_mode

        self.lateral_convs = None
        if lateral_conv:
            self.lateral_convs = nn.ModuleList()
            for i in range(len(img_channels)):
                l_conv = ConvModule(img_channels[i],
                                    mid_channels,
                                    3,
                                    padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=self.act_cfg,
                                    inplace=False)
                self.lateral_convs.append(l_conv)
            self.img_transform = nn.Sequential(
                nn.Linear(mid_channels * len(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        else:
            self.img_transform = nn.Sequential(
                nn.Linear(sum(img_channels), out_channels),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
            )
        self.pts_transform = nn.Sequential(
            nn.Linear(pts_channels, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        )

        if self.fuse_out:
            self.fuse_conv = nn.Sequential(
                nn.Linear(mid_channels, out_channels),
                # For pts the BN is initialized differently by default
                # TODO: check whether this is necessary
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=False))

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Xavier', layer='Conv2d', distribution='uniform'),
                dict(type='Xavier', layer='Linear', distribution='uniform')
            ]

    def forward(self, img_feats: List[Tensor], pts: List[Tensor],
                pts_feats: Tensor, img_metas: List[dict]) -> Tensor:
        """Forward function.

        Args:
            img_feats (List[Tensor]): Image features.
            pts: (List[Tensor]): A batch of points with shape N x 3.
            pts_feats (Tensor): A tensor consist of point features of the
                total batch.
            img_metas (List[dict]): Meta information of images.

        Returns:
            Tensor: Fused features of each point.
        """
        img_pts = self.obtain_mlvl_feats(img_feats, pts, img_metas)
        img_pre_fuse = self.img_transform(img_pts)
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(pts_feats)

        fuse_out = img_pre_fuse + pts_pre_fuse
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out

    def obtain_mlvl_feats(self, img_feats: List[Tensor], pts: List[Tensor],
                          img_metas: List[dict]) -> Tensor:
        """Obtain multi-level features for each point.

        Args:
            img_feats (List[Tensor]): Multi-scale image features produced
                by image backbone in shape (N, C, H, W).
            pts (List[Tensor]): Points of each sample.
            img_metas (List[dict]): Meta information for each sample.

        Returns:
            Tensor: Corresponding image features of each point.
        """
        if self.lateral_convs is not None:
            img_ins = [
                lateral_conv(img_feats[i])
                for i, lateral_conv in zip(self.img_levels, self.lateral_convs)
            ]
        else:
            img_ins = img_feats
        img_feats_per_point = []
        # Sample multi-level features
        for i in range(len(img_metas)):
            mlvl_img_feats = []
            for level in range(len(self.img_levels)):
                mlvl_img_feats.append(
                    self.sample_single(img_ins[level][i:i + 1], pts[i][:, :3],
                                       img_metas[i]))
            mlvl_img_feats = torch.cat(mlvl_img_feats, dim=-1)
            img_feats_per_point.append(mlvl_img_feats)

        img_pts = torch.cat(img_feats_per_point, dim=0)
        return img_pts

    def sample_single(self, img_feats: Tensor, pts: Tensor,
                      img_meta: dict) -> Tensor:
        """Sample features from single level image feature map.

        Args:
            img_feats (Tensor): Image feature map in shape (1, C, H, W).
            pts (Tensor): Points of a single sample.
            img_meta (dict): Meta information of the single sample.

        Returns:
            Tensor: Single level image features of each point.
        """
        # TODO: image transformation also extracted
        img_scale_factor = (pts.new_tensor(img_meta['scale_factor'][:2])
                            if 'scale_factor' in img_meta.keys() else 1)
        img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
        img_crop_offset = (pts.new_tensor(img_meta['img_crop_offset'])
                           if 'img_crop_offset' in img_meta.keys() else 0)
        proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
        img_pts = point_sample(
            img_meta=img_meta,
            img_features=img_feats,
            points=pts,
            proj_mat=pts.new_tensor(proj_mat),
            coord_type=self.coord_type,
            img_scale_factor=img_scale_factor,
            img_crop_offset=img_crop_offset,
            img_flip=img_flip,
            img_pad_shape=img_meta['input_shape'][:2],
            img_shape=img_meta['img_shape'][:2],
            aligned=self.aligned,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners,
        )
        return img_pts


def voxel_sample(voxel_features: Tensor,
                 voxel_range: List[float],
                 voxel_size: List[float],
                 depth_samples: Tensor,
                 proj_mat: Tensor,
                 downsample_factor: int,
                 img_scale_factor: Tensor,
                 img_crop_offset: Tensor,
                 img_flip: bool,
                 img_pad_shape: Tuple[int],
                 img_shape: Tuple[int],
                 aligned: bool = True,
                 padding_mode: str = 'zeros',
                 align_corners: bool = True) -> Tensor:
    """Obtain image features using points.

    Args:
        voxel_features (Tensor): 1 x C x Nx x Ny x Nz voxel features.
        voxel_range (List[float]): The range of voxel features.
        voxel_size (List[float]): The voxel size of voxel features.
        depth_samples (Tensor): N depth samples in LiDAR coordinates.
        proj_mat (Tensor): ORIGINAL LiDAR2img projection matrix for N views.
        downsample_factor (int): The downsample factor in rescaling.
        img_scale_factor (Tensor): Scale factor with shape of
            (w_scale, h_scale).
        img_crop_offset (Tensor): Crop offset used to crop image during
            data augmentation with shape of (w_offset, h_offset).
        img_flip (bool): Whether the image is flipped.
        img_pad_shape (Tuple[int]): Int tuple indicates the h & w after
            padding. This is necessary to obtain features in feature map.
        img_shape (Tuple[int]): Int tuple indicates the h & w before padding
            after scaling. This is necessary for flipping coordinates.
        aligned (bool): Whether to use bilinear interpolation when
            sampling image features for each point. Defaults to True.
        padding_mode (str): Padding mode when padding values for
            features of out-of-image points. Defaults to 'zeros'.
        align_corners (bool): Whether to align corners when
            sampling image features for each point. Defaults to True.

    Returns:
        Tensor: 1xCxDxHxW frustum features sampled from voxel features.
    """
    # construct frustum grid
    device = voxel_features.device
    h, w = img_pad_shape
    h_out = round(h / downsample_factor)
    w_out = round(w / downsample_factor)
    ws = (torch.linspace(0, w_out - 1, w_out) * downsample_factor).to(device)
    hs = (torch.linspace(0, h_out - 1, h_out) * downsample_factor).to(device)
    depths = depth_samples[::downsample_factor]
    num_depths = len(depths)
    ds_3d, ys_3d, xs_3d = torch.meshgrid(depths, hs, ws)
    # grid: (D, H_out, W_out, 3) -> (D*H_out*W_out, 3)
    grid = torch.stack([xs_3d, ys_3d, ds_3d], dim=-1).view(-1, 3)
    # recover the coordinates in the canonical space
    # reverse order of augmentations: flip -> crop -> scale
    if img_flip:
        # by default we take it as horizontal flip
        # use img_shape before padding for flip
        ori_h, ori_w = img_shape
        grid[:, 0] = ori_w - grid[:, 0]
    grid[:, :2] += img_crop_offset
    grid[:, :2] /= img_scale_factor
    # grid3d: (D*H_out*W_out, 3) in LiDAR coordinate system
    grid3d = points_img2cam(grid, proj_mat)
    # convert the 3D point coordinates to voxel coordinates
    voxel_range = torch.tensor(voxel_range).to(device).view(1, 6)
    voxel_size = torch.tensor(voxel_size).to(device).view(1, 3)
    # suppose the voxel grid is generated with AlignedAnchorGenerator
    # -0.5 given each grid is located at the center of the grid
    # TODO: study whether here needs -0.5
    grid3d = (grid3d - voxel_range[:, :3]) / voxel_size - 0.5
    grid_size = (voxel_range[:, 3:] - voxel_range[:, :3]) / voxel_size
    # normalize grid3d to (-1, 1)
    grid3d = grid3d / grid_size * 2 - 1
    # (x, y, z) -> (z, y, x) for grid_sampling
    grid3d = grid3d.view(1, num_depths, h_out, w_out, 3)[..., [2, 1, 0]]
    # align_corner=True provides higher performance
    mode = 'bilinear' if aligned else 'nearest'
    frustum_features = F.grid_sample(
        voxel_features,
        grid3d,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners)  # 1xCxDxHxW feats

    return frustum_features
