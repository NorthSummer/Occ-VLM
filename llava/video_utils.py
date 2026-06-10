import os
import json
import torch
import pickle
import cv2
import numpy as np
from PIL import Image
from transformers.image_utils import to_numpy_array
import json
from tqdm import tqdm
import random
import copy

def convert_from_uvd(u, v, d, intr, pose):
    # extr = np.linalg.inv(pose)
    
    fx = intr[0, 0]
    fy = intr[1, 1]
    cx = intr[0, 2]
    cy = intr[1, 2]
    depth_scale = 1000
    
    z = d / depth_scale
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    world = (pose @ np.array([x, y, z, 1]))
    return world[:3] / world[3]
    
def load_matrix_from_txt(path, shape=(4, 4)):
    with open(path) as f:
        txt = f.readlines()
    txt = ''.join(txt).replace('\n', ' ')
    matrix = [float(v) for v in txt.split()]
    return np.array(matrix).reshape(shape)


def unproject(intrinsics, poses, depths):
    """
        intrinsics: (V, 4, 4)
        poses: (V, 4, 4)
        depths: (V, H, W)
    """
    V, H, W = depths.shape
    y = torch.arange(0, H).to(depths.device)
    x = torch.arange(0, W).to(depths.device)
    y, x = torch.meshgrid(y, x)

    x = x.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)
    y = y.unsqueeze(0).repeat(V, 1, 1).view(V, H*W)     # (V, H*W)

    fx = intrinsics[:, 0, 0].unsqueeze(-1).repeat(1, H*W)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).repeat(1, H*W)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).repeat(1, H*W)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).repeat(1, H*W)

    z = depths.view(V, H*W) / 1000       # (V, H*W)
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    cam_coords = torch.stack([
        x, y, z, torch.ones_like(x)
    ], -1)      # (V, H*W, 4)

    world_coords = (poses @ cam_coords.permute(0, 2, 1)).permute(0, 2, 1)       # (V, H*W, 4)
    world_coords = world_coords[..., :3] / world_coords[..., 3].unsqueeze(-1)   # (V, H*W, 3)
    world_coords = world_coords.view(V, H, W, 3)

    return world_coords


class VideoProcessor:
    def __init__(
        self, 
        video_folder="data", 
        annotation_dir="data/embodiedscan/",
        voxel_size=None,
        min_xyz_range=None,
        max_xyz_range=None,
        frame_sampling_strategy='uniform',
        val_box_type='pred',
    ):
        self.video_folder = video_folder
        self.voxel_size = voxel_size
        self.min_xyz_range = torch.tensor(min_xyz_range) if min_xyz_range is not None else None
        self.max_xyz_range = torch.tensor(max_xyz_range) if max_xyz_range is not None else None
        self.frame_sampling_strategy = frame_sampling_strategy
        self.scene = {}
        print('============frame sampling strategy: {}============='.format(self.frame_sampling_strategy))

        for split in ["train", "val", "test"]:
            with open(os.path.join(annotation_dir, f"embodiedscan_infos_{split}.pkl"), "rb") as f:
                data = pickle.load(f)["data_list"]
                for item in data:
                    # item["sample_idx"]: "scannet/scene0415_00"
                    if item["sample_idx"].startswith("scannet"):
                        self.scene[item["sample_idx"]] = item

        self.scan2obj = {}

        for split in ['train', 'val']:
            box_type = "gt" if split == "train" else val_box_type
            filename = os.path.join("data", "metadata", f"scannet_{split}_{box_type}_box.json")
            with open(filename) as f:
                data = json.load(f)
                self.scan2obj.update(data)


        if 'mc' in self.frame_sampling_strategy:
            sampling_file = "data/metadata/scannet_select_frames.json"
            self.mc_sampling_files = {}
            with open(sampling_file) as f:
                data = json.load(f)
                for dd in data:
                    self.mc_sampling_files[dd['video_id']] = dd

            with open('data/metadata/pcd_discrete_0.1.pkl', 'rb') as f:
                pc_data = pickle.load(f)
            self.pc_min = {}
            self.pc_max = {}
            for scene_id in pc_data:
                pc_points = pc_data[scene_id]
                min_xyz = [1000, 1000, 1000]
                max_xyz = [-1000, -1000, -1000]
                for data in pc_points:
                    min_xyz = [min(v1, v2) for v1, v2 in zip(min_xyz, data)]
                    max_xyz = [max(v1, v2) for v1, v2 in zip(max_xyz, data)]
                self.pc_min[scene_id] = torch.Tensor(min_xyz) / 10
                self.pc_max[scene_id] = torch.Tensor(max_xyz) / 10


    def parse_data_info(self, info: dict, sample_indices) -> dict:
        """Process the raw data info.
        (This method is taken from EmbodiedScanDataset.parse_data_info with only
        path-prefix replaced by self.video_folder so it matches VideoProcessor.)
        """
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset != 'scannet':
            return None  # Skip non-ScanNet samples

        # expected attributes: self.box_type_3d, self.filter_empty_gt, etc.
        info['box_type_3d'] = getattr(self, 'box_type_3d', None)
        info['axis_align_matrix'] = self._get_axis_align_matrix(info)
        info['scan_id'] = info['sample_idx']
        ann_dataset = info['sample_idx'].split('/')[0]
        if ann_dataset == 'matterport3d':
            info['depth_shift'] = 4000.0
        else:
            info['depth_shift'] = 1000.0

        # Because multi-view settings are different from original designs
        # we temporarily follow the original design in ImVoxelNet
        info['img_path'] = []
        info['depth_img_path'] = []
        if 'cam2img' in info:
            cam2img = info['cam2img'].astype(np.float32)
        else:
            cam2img = []

        extrinsics = []
        for i in sample_indices: #range(len(info['images'])):
            # use self.video_folder as the prefix in VideoProcessor
            img_path = os.path.join(self.video_folder, info['images'][i]['img_path'])
            depth_img_path = os.path.join(self.video_folder, info['images'][i]['depth_path'])

            info['img_path'].append(img_path)
            info['depth_img_path'].append(depth_img_path)
            align_global2cam = np.linalg.inv(
                info['axis_align_matrix'] @ info['images'][i]['cam2global'])
            extrinsics.append(align_global2cam.astype(np.float32))
            if 'cam2img' not in info:
                cam2img.append(info['images'][i]['cam2img'].astype(np.float32))

        info['depth2img'] = dict(extrinsic=extrinsics,
                                 intrinsic=cam2img,
                                 origin=np.array([.0, .0, .5]).astype(np.float32))

        if 'depth_cam2img' not in info:
            info['depth_cam2img'] = cam2img


        return info

    @staticmethod
    def _get_axis_align_matrix(info: dict) -> np.ndarray:
        """Get axis_align_matrix from info. If not exist, return identity mat."""
        if 'axis_align_matrix' in info:
            return np.array(info['axis_align_matrix'])
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)


    def sample_frame_files_mc(self, video_id: str, frames_upbound: int = 32, do_shift=False):
        mc_files = self.mc_sampling_files[video_id]
        frame_files = mc_files['frame_files'][:frames_upbound]
        voxel_nums = mc_files['voxel_nums'][:frames_upbound]

        ratio = 1.0
        if 'ratio95' in self.frame_sampling_strategy:
            ratio = 0.95
        elif 'ratio90' in self.frame_sampling_strategy:
            ratio = 0.9

        if ratio != 1.0:
            num_all_voxels = mc_files['num_all_voxels']
            out = []
            cc = 0
            for frame_file, voxel_num in zip(frame_files, voxel_nums):
                out.append(frame_file)
                cc += voxel_num
                if cc >= num_all_voxels * ratio:
                    break
            frame_files = out

        frame_files.sort(key=lambda file: int(file.split('/')[-1].split('.')[0]))
        # if do_shift:
        #     ori_len = len(frame_files)
        #     i = random.randint(0, len(frame_files)-1)
        #     frame_files = frame_files[-i:] + frame_files[:-i]
        #     assert len(frame_files) == ori_len
        return frame_files  


    def sample_frame_files(
        self,
        video_id: str,
        force_sample: bool = False,
        frames_upbound: int = 0,
    ):
        # video_file: scannet/scene00000_01

        # since the color images have the suffix .jpg
        # frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f)) and os.path.join(video_file, f).endswith(".jpg")]
        # frame_files.sort()  # Ensure the frames are sorted if they are named sequentially
        meta_info = self.scene[video_id]
        frame_files = [os.path.join(self.video_folder, img["img_path"]) for img in meta_info["images"]]

        # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
        if force_sample:
            num_frames_to_sample = frames_upbound
        else:
            num_frames_to_sample = 10

        # For scannet, the RGB camera data is temporally synchronized with the depth sensor via hardware, providing synchronized depth and color capture at 30Hz
        # We follow embodiedscan by sampling one out of every ten images.
        avg_fps = 3
        
        total_frames = len(frame_files)
        sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        # frame_time = [i/3 for i in sampled_indices]
        # frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

        # video_time = total_frames / avg_fps

        return [frame_files[i] for i in sampled_indices], sampled_indices

    def calculate_world_coords(
        self,
        video_id: str, 
        frame_files,
        do_normalize=False,
    ):

        meta_info = self.scene[video_id]
        scene_id = video_id.split('/')[-1]

        axis_align_matrix = torch.from_numpy(np.array(meta_info['axis_align_matrix']))
        depth_intrinsic = torch.from_numpy(np.array(meta_info["depth_cam2img"]))

        depths = []
        poses = []
 
        # Read and store the sampled frames
        for frame_path in frame_files:

            # depth image
            depth_path = frame_path.replace(".jpg", ".png")
            with Image.open(depth_path) as depth_img:
                depth = np.array(depth_img).astype(np.int32)
                depths.append(torch.from_numpy(depth))

            # pose
            pose_file = frame_path.replace("jpg", "txt")
            pose = np.loadtxt(pose_file)
            poses.append(torch.from_numpy(pose))


        depths = torch.stack(depths)   # (V, H, W)
        poses = torch.stack([axis_align_matrix @ pose for pose in poses])     # (V, 4, 4)
        depth_intrinsic = depth_intrinsic.unsqueeze(0).repeat(len(frame_files), 1, 1)
        
        world_coords = unproject(depth_intrinsic.float(), poses.float(), depths.float())    # (V, H, W, 3)

        if do_normalize:
            world_coords = torch.maximum(world_coords, self.pc_min[scene_id].to(world_coords.device))
            world_coords = torch.minimum(world_coords, self.pc_max[scene_id].to(world_coords.device))
        
        return {
            "world_coords": world_coords,
        }

            

    def preprocess(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
    ):
        # 1. 采样帧和索引 (延续上一步的修改)
        if 'mc' in self.frame_sampling_strategy:
            frame_files, sampled_indices = self.sample_frame_files_mc(
                video_id,
                frames_upbound=frames_upbound,
                do_shift=('shift' in self.frame_sampling_strategy),
            )
        else:
            frame_files, sampled_indices = self.sample_frame_files(
                video_id,
                force_sample=force_sample,
                frames_upbound=frames_upbound,
            )

        # 2. 对元数据进行同步采样 (延续上一步的修改)
        raw_meta = self.scene[video_id] 
        parsed_meta = self.parse_data_info(raw_meta, sample_indices=sampled_indices)

        # 3. 计算/获取世界坐标 (此时 world_coords 已经是采样后的)
        video_dict = self.calculate_world_coords(
            video_id,
            frame_files,
            do_normalize=('norm' in self.frame_sampling_strategy),
        )
        world_coords = video_dict["world_coords"]
        V, H, W, _ = world_coords.shape
        
        world_coords_flat = world_coords.reshape(-1, 3)
        x_min, x_max = world_coords_flat[:, 0].min().item(), world_coords_flat[:, 0].max().item()
        y_min, y_max = world_coords_flat[:, 1].min().item(), world_coords_flat[:, 1].max().item()
        z_min, z_max = world_coords_flat[:, 2].min().item(), world_coords_flat[:, 2].max().item()
        boundry = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

        # 4. 加载图像并增加尺寸断言
        images = []
        original_sizes = []
        for frame_file in frame_files:
            with Image.open(frame_file) as img:
                frame = img.convert("RGB")
                images.append(frame)
                original_sizes.append(frame.size) # PIL.size 为 (width, height)

        # 断言所有图像的长宽都相同
        assert all(s == original_sizes[0] for s in original_sizes), \
            f"Video {video_id} contains images with inconsistent sizes: {set(original_sizes)}"
        
        img_w, img_h = original_sizes[0]

        # 5. 计算缩放比例 (scale_factor) 并执行 Resize/Crop
        crop_size = image_processor.crop_size["width"]
        w_scale, h_scale = 1.0, 1.0 # 默认为 1.0

        if strategy == "resize":
            # 这种情况下，直接强制缩放到 (crop_size, crop_size)
            images = [frame.resize((crop_size, crop_size)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (crop_size, crop_size), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            
            # 计算缩放因子 (参照你提供的代码逻辑)
            w_scale = crop_size / img_w
            h_scale = crop_size / img_h

        elif strategy == "center_crop":
            # 首先按比例缩放高度到 crop_size
            new_height = crop_size
            new_width = int(img_w * (crop_size / img_h))
            
            # 计算缩放阶段的因子
            w_scale = new_width / img_w
            h_scale = new_height / img_h # 即 crop_size / img_h
            
            images = [frame.resize((new_width, new_height)) for frame in images]
            resized_coords = [cv2.resize(coords.numpy(), (new_width, new_height), interpolation=cv2.INTER_NEAREST) for coords in world_coords]
            
            # 计算 crop 的边距
            left = (new_width - crop_size) // 2
            right = left + crop_size
            top = (new_height - crop_size) // 2
            bottom = top + crop_size
            
            images = [frame.crop((left, top, right, bottom)) for frame in images]
            resized_coords = [coords[top:bottom, left:right, :] for coords in resized_coords]
        
        # 封装结果
        return {
            "images": images,
            "world_coords": torch.from_numpy(np.stack(resized_coords)),
            "video_size": len(images),
            "boundry": boundry,
            "objects": torch.tensor(self.scan2obj[video_id]),
            "data_meta_info": parsed_meta,
            "scale_factor": (w_scale, h_scale), # 返回计算出的 scale_factor
            "img_shape": (images[0].height, images[0].width), # resize 后的形状
            "ori_shape": (img_h, img_w), # 原始形状
        }


    def process_3d_video(
        self,
        video_id: str, 
        image_processor,
        force_sample: bool = False,
        frames_upbound: int = 0,
        strategy: str = "center_crop",
    ):
        video_dict = self.preprocess(
            video_id,
            image_processor,
            force_sample,
            frames_upbound,
            strategy,
        )
        video_dict["images"] = image_processor.preprocess(video_dict["images"], return_tensors="pt")["pixel_values"]
        return video_dict

    
    def discrete_point(self, xyz):
        xyz = torch.tensor(xyz)
        if self.min_xyz_range is not None:
            xyz = torch.maximum(xyz, self.min_xyz_range.to(xyz.device))
        if self.max_xyz_range is not None:
            xyz = torch.minimum(xyz, self.max_xyz_range.to(xyz.device))
        if self.min_xyz_range is not None:
            xyz = (xyz - self.min_xyz_range.to(xyz.device)) 
            
        xyz = xyz / self.voxel_size
        return xyz.round().int().tolist()
    

def merge_video_dict(video_dict_list):
    new_video_dict = {}
    new_video_dict['box_input'] = []
    for k in video_dict_list[0]:
        if k in ["world_coords", 'images', 'objects']:
            new_video_dict[k] = torch.stack([video_dict[k] for video_dict in video_dict_list])
        elif k in ['box_input']:
            for video_dict in video_dict_list:
                if video_dict[k] is not None:
                    new_video_dict['box_input'].append(video_dict[k])
        elif k == 'data_meta_info' or k =="scale_factor" or k=="img_shape" or k=="ori_shape":
            new_video_dict[k] = [video_dict[k] for video_dict in video_dict_list]


    new_video_dict['box_input'] = torch.Tensor(new_video_dict['box_input'])
    return new_video_dict

