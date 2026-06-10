#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import math
import re
import time
import torch
import torch.nn as nn
from .multimodal_encoder.builder import build_vision_tower, build_occ_tower
from .multimodal_resampler.builder import build_vision_resampler
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from llava.utils import rank0_print, rank_print
import random


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            # self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.occ_tower = build_occ_tower()
            self.vision_resampler = build_vision_resampler(config, vision_tower=self.occ_tower.adapter.model.vision_tower)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.occ_tower.adapter.model.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
        
        if hasattr(self.config, 'world_position_embedding_type'):
            from llava.model.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingMLP

            if "sample9" in self.config.world_position_embedding_type:
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                n_points = 5
            elif "minmax" in self.config.world_position_embedding_type:
                n_points = 2
            else:
                n_points = 1
        
            if "mlp" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingMLP(config.hidden_size, n_points=n_points)
            elif "sin3d" in self.config.world_position_embedding_type:
                self.world_position_embedding = PositionEmbeddingSine3D(config.hidden_size, n_points=n_points)
            # elif "slp" in self.config.world_position_embedding_type:
            #     self.world_position_embedding = PositionEmbeddingSine3DMLP(config.hidden_size, n_points=n_points)
            

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_occ_tower(self):
        occ_tower = getattr(self, "occ_tower", None)
        if type(occ_tower) is list:
            occ_tower = occ_tower[0]
        return occ_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = "occ_tower" # model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        # ✅ 关键：检查是否已构建 occ_tower
        if self.get_occ_tower() is None:
            # 调用你自定义的 build_occ_tower 函数
            occ_tower = build_occ_tower() 
            vision_resampler = build_vision_resampler(model_args, vision_tower=occ_tower)

            # Save config from resampler
            for k, v in vision_resampler.config.items():
                setattr(self.config, k, v)

            # Handle FSDP wrapping
            if fsdp is not None and len(fsdp) > 0:
                self.occ_tower = [occ_tower]
                self.vision_resampler = [vision_resampler]
            else:
                self.occ_tower = occ_tower
                self.vision_resampler = vision_resampler
        else:
            # Already exists, just retrieve
            if fsdp is not None and len(fsdp) > 0:
                occ_tower = self.occ_tower[0]
                vision_resampler = self.vision_resampler[0]
            else:
                occ_tower = self.occ_tower
                vision_resampler = self.vision_resampler

            vision_resampler = self.vision_resampler
            vision_tower = self.occ_tower
            vision_tower.load_model()
     
        # for p in self.vision_resampler.parameters():
        #     p.requires_grad = True

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_hidden_size = getattr(vision_resampler, "hidden_size", vision_tower.adapter.model.hidden_size)
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        
        if not hasattr(self.config, 'add_faster_video'):
            if model_args.add_faster_video:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.faster_token = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.adapter.model.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            rank0_print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")
            incompatible_keys = self.vision_resampler.load_state_dict(get_w(mm_projector_weights, "vision_resampler"), strict=False)
            rank0_print(f"Loaded vision resampler weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of the image (height, width).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    # Compute aspect ratios
    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    # Determine padding size and direction
    if original_aspect_ratio > current_aspect_ratio:
        # Padding was added to the height
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # Padding was added to the width
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_occ_tower(self):
        return self.get_model().get_occ_tower()

    def get_2dPool(self, image_feature, stride=2):
        height = width = self.get_vision_tower().num_patches_per_side
        num_frames, num_tokens, num_dim = image_feature.shape
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            image_feature = nn.functional.avg_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "max":
            image_feature = nn.functional.max_pool2d(image_feature, stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, width = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / stride), math.ceil(width / stride)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def average_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]
        world_coords_avg = torch.nn.functional.avg_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        patch_num = world_coords_avg.shape[-1]
        world_coords_avg = world_coords_avg.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        return world_coords_avg

    def minmax_coordinate_in_patch(self, world_coords, patch_size=27):

        V, H, W, D = world_coords.size() # D = 3

        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :]    # [32, 378, 378, 3]
        world_coords = world_coords.permute(0, 3, 1, 2)   # [V, D, 378, 378]

        world_coords_max = torch.nn.functional.max_pool2d(world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_max = world_coords_max.permute(0, 2, 3, 1)     # [32, 14, 14, 3]

        world_coords_min = - torch.nn.functional.max_pool2d(-world_coords, kernel_size=patch_size, stride=patch_size)  # [32, 3, 14,  14]
        world_coords_min = world_coords_min.permute(0, 2, 3, 1)     # [32, 14, 14, 3]
        world_coords = torch.stack([world_coords_min, world_coords_max], dim=3) # [32, 14, 14, 2, 3]

        return world_coords
    
    def sample_n_points(self, world_coords, n_points=9):

        V, H, W, D = world_coords.size() # D = 3
        world_coords = world_coords.view(V, H, W, D)[:, :-6, :-6, :] 
        world_coords = world_coords.view(-1, 14, 27, 14, 27, 3).permute(0, 1, 3, 2, 4, 5)
        if n_points == 9:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
        elif n_points == 5:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 0::2, :].reshape(V, 14, 14, 5, 3)
        elif n_points == 1:
            world_coords_sample = world_coords[:, :, :, 4::9, 4::9, :].reshape(V, 14, 14, 9, 3)
            world_coords_sample = world_coords_sample[:, :, :, 4, :].reshape(V, 14, 14, 3)
        else:
            raise NotImplementedError
        
        return world_coords_sample

    def discrete_coords(self, world_coords):

        # V, H, W, D = world_coords.size() # D = 3
        # world_coords_discrete = (world_coords.view(-1, 3) - xyz_min.view(1, 3)) / self.config.voxel_size

        min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        max_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)

        voxel_size = 0.16

        #min_xyz_range = torch.tensor([-6.4, -6.4, -1.56], device=device, dtype=dtype)
        #max_xyz_range = torch.tensor([6.4, 6.4, 3.56], device=device, dtype=dtype)

        world_coords = torch.maximum(world_coords, min_xyz_range)
        world_coords = torch.minimum(world_coords, max_xyz_range)
        world_coords_discrete = (world_coords - min_xyz_range) / voxel_size # self.config.voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach().unsqueeze(0)


    def discrete_occ_coords(self, world_coords):

        device = world_coords.device
        dtype = world_coords.dtype

        min_xyz_range = torch.tensor([-6.4, -6.4, -1.56], device=device, dtype=dtype)
        max_xyz_range = torch.tensor([6.4, 6.4, 3.56], device=device, dtype=dtype)

        # min_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)
        # max_xyz_range = torch.tensor(self.config.min_xyz_range).to(world_coords.device)

        voxel_size = 0.16

        clamped_coords = torch.clamp(world_coords, min=min_xyz_range, max=max_xyz_range)
        world_coords_discrete = (clamped_coords - min_xyz_range) / voxel_size
        world_coords_discrete = world_coords_discrete.round()

        return world_coords_discrete.detach().unsqueeze(0)


    def encode_images(self, images, world_coords=None):
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        image_features = self.get_model().mm_projector(image_features) # torch.Size([32, 729, 1152])

        return image_features

    def calculate_occ_coords(self, occ_grid, 
        scene_range = [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78], 
        grid_size = 0.16):

        x_min, y_min, z_min, x_max, y_max, z_max = scene_range
        B = occ_grid.shape[0]
        batch_centers = []

        for b in range(B):
            occ = occ_grid[b]
            indices = torch.nonzero(occ, as_tuple=False)
            if indices.numel() == 0:
                batch_centers.append(torch.empty((0, 3), device=occ_grid.device))
                continue
            ix = indices[:, 0].float()
            iy = indices[:, 1].float()
            iz = indices[:, 2].float()

            x = x_min + (ix + 0.5) * grid_size 
            y = y_min + (iy + 0.5) * grid_size 
            z = z_min + (iz + 0.5) * grid_size 

            centers = torch.stack([x, y, z], dim=-1)
            batch_centers.append(centers)
        batch_centers = torch.cat(batch_centers, dim=0) 
        
        return batch_centers

    def encode_images_with_occ(self, images, video_dict=None, object_boxes_center=None):
        occ_pred, vfm_last_hidden = self.get_model().get_occ_tower()(images, video_dict, mode="predict")
        
        assert type(occ_pred) == list
        occ_grid_mapping = {
            1: 0.16,
            2: 0.32,
            3: 0.48
        }
        occ_level = 1
        grid_size = occ_grid_mapping.get(occ_level, 0.16)

        occ_coords = self.calculate_occ_coords(occ_pred[occ_level-1], grid_size = 0.16)
        if object_boxes_center is not None:
            occ_coords = torch.cat([occ_coords, object_boxes_center], dim=0)

        occ_features = self.get_model().get_occ_tower().resample_feat(vfm_last_hidden, occ_coords, video_dict)
        # print(f"occ_feat_shape: {occ_features.shape}")
        visual_length = occ_features.shape[1]

        # with open("visual_length_sqa3d_level_3.txt", "a") as f:
        #     f.write(f"{visual_length}\n")

        occ_features = self.get_model().mm_projector(occ_features)
    
        return occ_features, occ_coords, vfm_last_hidden

    def encode_multimodals(self, videos_or_images, video_idx_in_batch, split_sizes=None):
        videos_or_images_features = self.get_model().get_vision_tower()(videos_or_images)
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []
        all_faster_video_features = []
        cur_mm_spatial_pool_stride = self.config.mm_spatial_pool_stride

        for idx, feat in enumerate(per_videos_or_images_features):
            
            feat = self.get_model().mm_projector(feat)
            faster_video_feature = 0
            slower_img_feat = 0
            if idx in video_idx_in_batch and cur_mm_spatial_pool_stride > 1:
                slower_img_feat = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
                if self.config.add_faster_video:
                    cur_mm_spatial_pool_stride = cur_mm_spatial_pool_stride * 2
                    faster_video_feature = self.get_2dPool(feat,cur_mm_spatial_pool_stride)
            if slower_img_feat is not 0:
                all_videos_or_images_features.append(slower_img_feat)
            else:
                all_videos_or_images_features.append(feat)
            all_faster_video_features.append(faster_video_feature)
        return all_videos_or_images_features,all_faster_video_features

    def add_token_per_grid(self, image_feature):
        resize_h = int(math.sqrt(image_feature.shape[1]))
        num_frames = image_feature.shape[0]
        feature_dim = image_feature.shape[-1]

        image_feature = image_feature.view(num_frames, 1, resize_h, resize_h, -1)
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
        image_feature = torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        if getattr(self.config, "add_faster_video", False):
            # import pdb; pdb.set_trace()
            # (3584, 832, 14) -> (3584, 64, 13, 14)
            image_feature = image_feature.view(feature_dim, num_frames,resize_h, -1)
            #  (3584, 64, 13, 14) -> (64, 13, 14, 3584)
            image_feature = image_feature.permute(1, 2, 3, 0).contiguous()
            # (64, 13, 14, 3584) -> (64, 13*14, 3584)
            image_feature = image_feature.flatten(1, 2)
            # import pdb; pdb.set_trace()
            return image_feature
        # import pdb; pdb.set_trace()
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
        return image_feature

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature


    def prepare_inputs_labels_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
    ):

        object_boxes = None
        object_boxes_center_ori = None

        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_boxes_center_ori = object_boxes_center
            object_features = []
            obj_num = len(object_boxes)

        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center)

            if 'mrope' in self.config.world_position_embedding_type:
                use_mrope_position_embedding = True
            
            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True
            
            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        # vision_tower = self.get_vision_tower()
        occ_tower = self.get_occ_tower()

        # # 统计解冻（需要梯度）和冻结的参数数量
        # trainable_params = sum(p.numel() for p in occ_tower.parameters() if p.requires_grad)
        # total_params = sum(p.numel() for p in occ_tower.parameters())

        # print(f"--- [occ_tower 状态检查] ---")
        # print(f"总参数量: {total_params:,}")
        # print(f"待训练(已解冻)参数量: {trainable_params:,}")
        # print(f"解冻比例: {100 * trainable_params / total_params:.2f}%")
        # print(f"是否完全冻结: {'是' if trainable_params == 0 else '否'}")

        # import pdb
        # pdb.set_trace()

        if occ_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        # rank_print(modalities)
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            object_boxes_center_ori = None
            encoded_occ_features, occ_coords, vfm_last_hidden = self.encode_images_with_occ(concat_images, video_dict, object_boxes_center_ori)
            
       
            if use_object_proposals:
                assert object_boxes is not None
                obj_num = len(object_boxes)
                object_patch = []
                voxel_size_half = 0.08

                for l in range(obj_num):
                    box = object_boxes[l]

                    box_min = box[:3] - box[3:] / 2
                    box_max = box[:3] + box[3:] / 2
                    
                    box_min = box_min.to(occ_coords.device)
                    box_max = box_max.to(occ_coords.device)

                    voxel_min = occ_coords - voxel_size_half
                    voxel_max = occ_coords + voxel_size_half

                    overlap_mask = (voxel_min <= box_max) & (voxel_max >= box_min)
                    cur_object_mask = overlap_mask.all(dim=-1) # (N,)
                    
                    object_patch.append(cur_object_mask)

                object_features = []
                valid_obj_num = 0
                fallback_obj_num = 0 

                fallback_features_list = []
                fallback_coords_list = []
                # ==================== [新增结束] ====================
                
                for l in range(obj_num):
 
                    mask = object_patch[l] 
                    # [关键] 使用encoded_occ_features（不带PE），而非occ_features_with_pe
                    cur_object_features = encoded_occ_features[0][mask]

                    if cur_object_features.shape[0] == 0:
  
                        # print(f"[Fallback] Object box {l} has no overlapping voxels, resampling from VFM")
                        
                        # 获取object中心坐标 (1, 3)
                        box_center = object_boxes[l][:3].unsqueeze(0).to(occ_coords.device)
                        
                        resampled_feat = self.get_model().get_occ_tower().resample_feat(
                            vfm_last_hidden, 
                            box_center.float(), 
                            video_dict
                        )
                        resampled_feat = self.get_model().mm_projector(resampled_feat)

                        # resampled_feat: (1, C)
                        cur_object_features = resampled_feat.squeeze(0).squeeze(0)  # (C,)
                        
                        # 检查是否为零向量（即无效点）
                        is_valid = cur_object_features.abs().sum() > 1e-6
                        
                        if is_valid:
                            # 将采样到的特征和坐标加入列表，后续cat到occ_features
                            fallback_features_list.append(resampled_feat.squeeze(0))  # (1, C)
                            fallback_coords_list.append(box_center)        # (1, 3)
                            valid_obj_num += 1
                            print(f"[Fallback] Object box {l} resampled successfully")
                        else:
                            # 无效点（不在图像区域内），保持零向量
                            print(f"[Fallback] Object box {l} is out of image, using zero vector")
                        
                        fallback_obj_num += 1
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1           
                        # print(cur_object_features.shape, "valid_shape")
               
                    object_features.append(cur_object_features)
           
                object_features = torch.stack(object_features)

                if len(fallback_features_list) > 0:
                    fallback_features = torch.cat(fallback_features_list, dim=0)  # (M, C)
                    fallback_coords = torch.cat(fallback_coords_list, dim=0)      # (M, 3)
                    # print(fallback_features.shape, fallback_coords.shape, 555)
                    # cat到encoded_occ_features和occ_coords
                    # 注意：假设batch_size=1，如果batch_size>1需要相应调整
                    # print(encoded_occ_features.shape, fallback_features)
                    encoded_occ_features = torch.cat(
                        [encoded_occ_features, fallback_features.unsqueeze(0)], dim=1
                    )  # (1, N+M, C)
                    # print(occ_coords.shape, fallback_coords.shape, 777)
                    occ_coords = torch.cat(
                        [occ_coords, fallback_coords], dim=0
                    ).unsqueeze(0)  # (1, N+M, 3)
                    
                    # print(f"[Fallback] Added {len(fallback_features_list)} resampled features to occ_features, "
                    #       f"new occ_features shape: {encoded_occ_features.shape}")
                # print(object_boxes_center.shape, 888)
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center).squeeze(0)      
                    object_features += box_center_features

                # print(f"Object stats: valid={valid_obj_num}, fallback={fallback_obj_num}, total={obj_num}")
            else:
                object_features = None


            assert use_sin3d_pe or use_mlp_pe, "Expected use_sin3d_pe or use_mlp_pe to be True"
            if use_sin3d_pe or use_mlp_pe:        
                if "discrete" in self.config.world_position_embedding_type:        
                    occ_coords_discrete = self.discrete_coords(occ_coords.squeeze(0))
                # print(occ_coords_discrete.shape, 999)
                occ_feat = encoded_occ_features + self.get_model().world_position_embedding(occ_coords_discrete.detach())
                occ_features = occ_feat.to(encoded_occ_features.dtype)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images) tensor(1, device='cuda:0')
            if num_images == 0:
                cur_image_features = encoded_occ_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])
                if query_coord_tokens.sum() != 0:
                    cur_input_embeds = cur_input_embeds.clone()
                    cur_input_embeds[query_coord_tokens] += self.get_model().world_position_embedding(box_input.detach())[:, 0]
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    try:
                        cur_occ_features = occ_features[cur_image_idx]
                    except IndexError:
                        cur_occ_features = occ_features[cur_image_idx - 1]
                    
                    cur_image_idx += 1
                    # cur_new_input_embeds.append(cur_image_features)
                    # cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    cur_new_input_embeds.append(cur_occ_features)
                    cur_new_labels.append(torch.full((cur_occ_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes




    def prepare_inputs_labels_for_multimodal_(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels, 
        images, 
        modalities=["image"], 
        image_sizes=None, 
        video_dict=None,
        use_object_proposals: bool = False,
    ):

        object_boxes = None
        object_boxes_center_ori = None

        if use_object_proposals:
            object_boxes = video_dict["objects"][0]
            object_boxes_center = object_boxes[:, :3]
            object_boxes_center_ori = object_boxes_center
            object_features = []
            obj_num = len(object_boxes)

        use_mrope_position_embedding = False
        use_sin3d_pe = False
        use_mlp_pe = False
        if hasattr(self.config, 'world_position_embedding_type') and past_key_values is None:
            B = input_ids.shape[0]
            world_coords = video_dict['world_coords']
            xyz_min = world_coords.view(B, -1, 3).min(dim=1)[0]

            if len(video_dict['box_input']):
                box_input = video_dict['box_input']     # [1, 3]
            else:
                box_input = None

            n_points = 1
            if 'avg' in self.config.world_position_embedding_type:
                world_coords = [self.average_coordinate_in_patch(coords) for coords in world_coords]
            elif "sample9" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=9) for coords in world_coords]
                n_points = 9
            elif "sample5" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=5) for coords in world_coords]
                n_points = 5
            elif "sample1" in self.config.world_position_embedding_type:
                world_coords = [self.sample_n_points(coords, n_points=1) for coords in world_coords]
            elif "minmax" in self.config.world_position_embedding_type:
                world_coords = [self.minmax_coordinate_in_patch(coords) for coords in world_coords]
                n_points = 2

            if n_points > 1:
                if box_input is not None:
                    box_input = box_input[:, None, :].repeat(1, n_points, 1)
                if object_boxes is not None:
                    object_boxes_center = object_boxes_center[:, None, :].repeat(1, n_points, 1)

            if 'discrete' in self.config.world_position_embedding_type or use_mrope_position_embedding:
                world_coords_discrete = [self.discrete_coords(coords, xyz_min[i]) for i, coords in enumerate(world_coords)]
                if box_input is not None:
                    box_input = self.discrete_coords(box_input, None)
                if object_boxes is not None:
                    object_boxes_center = self.discrete_coords(object_boxes_center, None)

            if 'mrope' in self.config.world_position_embedding_type:
                use_mrope_position_embedding = True
            
            if "sin3d" in self.config.world_position_embedding_type:
                use_sin3d_pe = True
            
            if "mlp" in self.config.world_position_embedding_type:
                use_mlp_pe = True


        # vision_tower = self.get_vision_tower()
        occ_tower = self.get_occ_tower()
        if occ_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        # rank_print(modalities)
        # if vision_tower is None or images is None or input_ids.shape[1] == 1:
        #     return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        if isinstance(modalities, str):
            modalities = [modalities]

        # import pdb; pdb.set_trace()
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))

            concat_images = torch.cat([image for image in images_list], dim=0)
            split_sizes = [image.shape[0] for image in images_list]

            object_boxes_center_ori = None
            encoded_occ_features, occ_coords = self.encode_images_with_occ(concat_images, video_dict, object_boxes_center_ori)
            
       
            if use_object_proposals:
                assert object_boxes is not None
                obj_num = len(object_boxes)
                object_patch = []
                voxel_size_half = 0.08

                for l in range(obj_num):
                    box = object_boxes[l]

                    # 1. 计算物体检测框 (Bounding Box) 的边界
                    box_min = box[:3] - box[3:] / 2
                    box_max = box[:3] + box[3:] / 2
                    
                    box_min = box_min.to(occ_coords.device)
                    box_max = box_max.to(occ_coords.device)

                    # 2. 计算所有 Voxel 的边界 (基于中心点 occ_coords 和 voxel_size)
                    # occ_coords shape: (N, 3)
                    voxel_min = occ_coords - voxel_size_half
                    voxel_max = occ_coords + voxel_size_half

                    # 3. 判断 AABB 重叠逻辑：
                    # 只有当 (VoxelMin <= BoxMax) 且 (VoxelMax >= BoxMin) 时，两个框在某个维度上才有重叠
                    # 我们需要在 X, Y, Z 三个维度上都满足这个条件
                    overlap_mask = (voxel_min <= box_max) & (voxel_max >= box_min)
                    cur_object_mask = overlap_mask.all(dim=-1) # (N,)
                    
                    object_patch.append(cur_object_mask)
            # 2
            # if use_object_proposals:
                object_features = []
                valid_obj_num = 0
                
                for l in range(obj_num):
 
                    mask = object_patch[l] 
                    cur_object_features = encoded_occ_features[0][mask]

                    if cur_object_features.shape[0] == 0:
                        print(f"Non-valid object box {l}: {object_boxes[l]}")
                        cur_object_features = torch.zeros(encoded_occ_features.shape[-1]).to(encoded_occ_features.device)
                    else:
                        cur_object_features = cur_object_features.mean(dim=0)
                        valid_obj_num += 1           
               
                    object_features.append(cur_object_features)
                object_features = torch.stack(object_features)
    
                if use_mlp_pe or use_sin3d_pe:
                    box_center_features = self.get_model().world_position_embedding(object_boxes_center.unsqueeze(0)).squeeze(0)      
                    object_features += box_center_features
                print(f"valid_obj_num: {valid_obj_num}")
            else:
                object_features = None

            # 3
            assert use_sin3d_pe or use_mlp_pe
            if use_sin3d_pe or use_mlp_pe:        
                if "discrete" in self.config.world_position_embedding_type:        
                    occ_coords_discrete = self.discrete_occ_coords(occ_coords) # self.discrete_coords(occ_coords, None) ## occ feature           
                occ_feat = encoded_occ_features + self.get_model().world_position_embedding(occ_coords_discrete.detach())
                # 看看world_position_embedding的逻辑是否适配Occ grid size
                occ_features = occ_feat.to(encoded_occ_features.dtype)

            # 4
        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank_print(f"Total images : {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        # 5
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_world_coords = []
        cur_image_idx = 0
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # rank0_print(num_images) tensor(1, device='cuda:0')
            if num_images == 0:
                cur_image_features = encoded_occ_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]

            cat_cur_input_ids_noim = torch.cat(cur_input_ids_noim)
            cur_input_embeds = self.get_model().embed_tokens(cat_cur_input_ids_noim)

            # Add input coord PE
            if hasattr(self.config, "coord_token_ids") and (use_sin3d_pe or use_mlp_pe):
                query_coord_tokens = (cat_cur_input_ids_noim == self.config.coord_token_ids[0])
                if query_coord_tokens.sum() != 0:
                    cur_input_embeds = cur_input_embeds.clone()
                    cur_input_embeds[query_coord_tokens] += self.get_model().world_position_embedding(box_input.unsqueeze(0).detach())[:, 0]
            
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_world_coords = []
            cur_pos_index = 0
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    try:
                        cur_occ_features = occ_features[cur_image_idx]
                    except IndexError:
                        cur_occ_features = occ_features[cur_image_idx - 1]
                    
                    cur_image_idx += 1
                    # cur_new_input_embeds.append(cur_image_features)
                    # cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

                    cur_new_input_embeds.append(cur_occ_features)
                    cur_new_labels.append(torch.full((cur_occ_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]
        # TODO: Hard code for control loss spike
        # if tokenizer_model_max_length is not None:
        #     new_input_embeds = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        #     new_labels = [x[:4096] if modality != "video" else x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        mrope_position_ids = torch.zeros((batch_size, max_len, 3), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        # mrope_position_ids = mrope_position_ids.permute(2, 0, 1)
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        
        if use_mrope_position_embedding:
            position_ids = mrope_position_ids

        # import pdb; pdb.set_trace()
        # rank0_print("Finish preparing")
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, object_features, object_boxes

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False