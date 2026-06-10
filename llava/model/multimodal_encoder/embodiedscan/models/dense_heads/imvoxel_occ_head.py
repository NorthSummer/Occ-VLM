# Copyright (c) OpenRobotLab. All rights reserved.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor


from llava.model.multimodal_encoder.embodiedscan.registry import MODELS
from llava.model.multimodal_encoder.embodiedscan.utils.typing_config import SampleList

import numpy as np
import os

def save_pred_occ_3d(pred_occ: torch.Tensor):
    """
    pred_occ: torch.Tensor, shape (1,X,Y,Z) or (X,Y,Z). labels in 0-255 (0 means empty).
    out_path: path to save the resulting (N,4) numpy array.
    Returns:
        occ_pred_save: numpy array shape (N,4) dtype=int32, rows (x,y,z,label)
    """
    # ensure tensor on CPU and detached
    pred = pred_occ.detach().cpu()
    # remove batch dim if present
    if pred.dim() == 4 and pred.size(0) == 1:
        pred = pred.squeeze(0)  # now (X,Y,Z)
    if pred.dim() != 3:
        raise ValueError(f"Expected pred_occ with 3 dims (X,Y,Z) after squeezing, got {pred.dim()} dims")

    # find non-zero voxels
    idx = torch.nonzero(pred != 0, as_tuple=False)  # (N,3) rows: (x,y,z)
    if idx.numel() == 0:
        occ_pred_save = np.zeros((0, 4), dtype=np.int32)
    else:
        labels = pred[idx[:, 0], idx[:, 1], idx[:, 2]].long().unsqueeze(1)  # (N,1)
        out = torch.cat([idx.long(), labels], dim=1)  # (N,4) tensor
        occ_pred_save = out.numpy().astype(np.int64)

    # save
    with open('/data/ljn/code/EmbodiedScan/last_scan.txt' , 'r', encoding='utf-8') as f:
        for line in f: 
            scene_id = line.strip() 

    out_path = f'/data/ljn/data/embodiedscan/Scannet/scans/{scene_id}/occupancy'
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    save_path = f'{out_path}/occupancy.npy'

    np.save(save_path, occ_pred_save)
    print(f"successful saved occ {out_path}")
    # return occ_pred_save


@MODELS.register_module()
class ImVoxelOccHead(BaseModule):
    """Occupancy prediction head compatible with ImVoxelNeck outputs.

    Args:
        num_classes (int): Number of categories. Defaults to 21.
        volume_h (int): Size along h of the 3D volume. Defaults to 40.
        volume_w (int): Size along w of the 3D volume. Defaults to 40.
        volume_z (int): Size along z of the 3D volume. Defaults to 16.
        in_channels (int): Input channels. Defaults to 128.
        use_semantic (bool): Whether to use semantic predictions.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 num_classes=21,
                 volume_h=40,
                 volume_w=40,
                 volume_z=16,
                 in_channels=128,
                 use_semantic=True,
                 **kwargs):
        super(ImVoxelOccHead, self).__init__()
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.in_channels = in_channels
        self.use_semantic = use_semantic

        self._init_layers()

    def _init_layers(self):
        conv_cfg = dict(type='Conv3d', bias=False)
        self.occ = nn.ModuleList()
        for i in range(len(self.in_channels)):
            if self.use_semantic:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=self.num_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=1,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)

    def forward(self, mlvl_feats):
        """Forward function.

        Args:
            mlvl_feats (list[Tensor]): Multi-level features.
            input_metas (list[dict]): Input meta infos.

        Returns:
            list[Tensor]: Occupancy predicted maps.
        """
        occ_preds = []
        for i in range(len(mlvl_feats)):
            occ_pred = self.occ[i](mlvl_feats[i])
            occ_preds.append(occ_pred)

        return occ_preds

    def predict(self, x: Tuple[Tensor]):
        """Predict/Inference function.

        Args:
            x (Tuple[Tensor]): Multi-level features.
            batch_data_samples (`SampleList`): Batch of data samples.

        Returns:
            Tensor: Occupancy predictions.
        """

        pred = self.forward(x)# [0]
        pred_list = []
        if self.use_semantic:
            for i in range(len(pred)):
                _, pred_occ = torch.max(torch.softmax(pred[i], dim=1), dim=1)
                pred_list.append(pred_occ)
        else:
            pred_list = torch.sigmoid(pred[:, 0])

        # save_pred_occ_3d(pred_occ)
        # import pdb
        # pdb.set_trace()
        return pred_list



