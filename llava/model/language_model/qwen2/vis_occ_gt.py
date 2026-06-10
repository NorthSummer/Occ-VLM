import open3d as o3d
import numpy as np
import os
import pickle
from PIL import Image  # 需要 pip install pillow
from tqdm import tqdm

class OccGtVisualizerStrict:
    def __init__(self, pkl_path, save_dir, voxel_size=0.16):
        self.save_dir = save_dir
        self.voxel_size = voxel_size
        
        # 1. 颜色表 (10种颜色)
        self.colors_palette = np.array([
            [255, 120,  50, 255], [255, 192, 203, 255], [255, 255,   0, 255],
            [  0, 150, 245, 255], [  0, 255, 255, 255], [0, 175,   0, 255],
            [255,   0,   0, 255], [127, 127, 127, 255], [135,  60,   0, 255],
            [160,  32, 240, 255],
        ]).astype(np.float64) / 255.0
        self.colors_palette = self.colors_palette[:, :3] 

        # 2. 初始化映射表
        self.occ_label_mapping = None
        self._init_mapping(pkl_path)

        self.point_cloud_range = np.array([-3.2, -3.2, -1.28 + 0.5, 3.2, 3.2, 1.28 + 0.5])

    def _init_mapping(self, pkl_path):
        print(f"Loading metainfo from {pkl_path} ...")
        with open(pkl_path, 'rb') as f:
            scene_pkg = pickle.load(f)
            categories = scene_pkg['metainfo']['categories']

        max_id = max(list(categories.values())) + 1
        self.occ_label_mapping = np.full(max_id, -1, dtype=int)

        occ_classes = (
            'floor', 'wall', 'chair', 'cabinet', 'door', 'table', 'couch',
            'shelf', 'window', 'bed', 'curtain', 'desk', 'doorframe',
            'plant', 'stairs', 'pillow', 'wardrobe', 'picture', 'bathtub',
            'box', 'counter', 'bench', 'stand', 'rail', 'sink', 'clothes',
            'mirror', 'toilet', 'refrigerator', 'lamp', 'book', 'dresser',
            'stool', 'fireplace', 'tv', 'blanket', 'commode',
            'washing machine', 'monitor', 'window frame', 'radiator', 'mat',
            'shower', 'rack', 'towel', 'ottoman', 'column', 'blinds',
            'stove', 'bar', 'pillar', 'bin', 'heater', 'clothes dryer',
            'backpack', 'blackboard', 'decoration', 'roof', 'bag', 'steps',
            'windowsill', 'cushion', 'carpet', 'copier', 'board',
            'countertop', 'basket', 'mailbox', 'kitchen island',
            'washbasin', 'bicycle', 'drawer', 'oven', 'piano',
            'excercise equipment', 'beam', 'partition', 'printer',
            'microwave', 'frame'
        )

        for idx, label_name in enumerate(occ_classes):
            if label_name in categories:
                self.occ_label_mapping[categories[label_name]] = idx + 1

    def process_and_save(self, npy_path, scan_id, region):
        save_path = os.path.join(self.save_dir, f"{scan_id}_gt_vis_{region}.png")
        
        try:
            gt_occ = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            return

        if gt_occ.shape[0] == 0: return

        # Mapping Logic
        raw_labels = gt_occ[:, 3].astype(int)
        valid_raw_mask = raw_labels < len(self.occ_label_mapping)
        gt_occ = gt_occ[valid_raw_mask]
        raw_labels = raw_labels[valid_raw_mask]
        
        mapped_labels = self.occ_label_mapping[raw_labels]
        valid_mapped_mask = mapped_labels > 0
        final_data = gt_occ[valid_mapped_mask]
        mapped_labels = mapped_labels[valid_mapped_mask]

        if len(final_data) == 0: return

        # Strict Color Filter
        color_indices = mapped_labels - 1
        color_valid_mask = color_indices < len(self.colors_palette)
        final_data = final_data[color_valid_mask]
        color_indices = color_indices[color_valid_mask]

        if len(final_data) == 0: return

        # Coords conversion
        grid_coords = final_data[:, :3]
        min_range = self.point_cloud_range[:3]
        points = grid_coords * self.voxel_size + min_range + (self.voxel_size / 2)
        point_colors = self.colors_palette[color_indices]

        # 渲染并保存为白底 PNG
        success = self._render_open3d(points, point_colors, save_path)
        
        # 后处理：去底
        if success:
            self._make_transparent(save_path)

    def _render_open3d(self, points, colors, save_path):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.voxel_size
        )

        try:
            width, height = 1920, 1080
            renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
            
            # 关键：设置为纯白色背景
            renderer.scene.set_background([1.0, 1.0, 1.0, 1.0]) 
            
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "defaultLit"
            renderer.scene.add_geometry("voxels", voxel_grid, mat)
            
            center = voxel_grid.get_center()
            offset = [6, -6, 6]
            renderer.setup_camera(60.0, center, center + offset, [0, 0, 1])
            
            img = renderer.render_to_image()
            o3d.io.write_image(save_path, img)
            return True
            
        except Exception as e:
            print(f"Open3D render failed: {e}")
            return False

    def _make_transparent(self, img_path):
        """
        使用 NumPy 加速处理：将接近白色的背景变为透明
        """
        try:
            img = Image.open(img_path).convert("RGBA")
            data = np.array(img) # 转为 NumPy 数组，速度极快

            # 分离通道
            r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

            # 设定阈值：RGB 均大于 240 则认为是背景 (处理抗锯齿边缘)
            # 你的颜色表里最亮的是 Yellow (255,255,0) 和 Cyan (0,255,255)
            # Yellow 的 B=0，Cyan 的 R=0，所以只要三通道都大于 240，绝对是背景白色
            threshold = 220 
            mask = (r > threshold) & (g > threshold) & (b > threshold)

            # 将满足条件的像素 Alpha 设为 0
            data[mask, 3] = 0

            # 保存回图片
            new_img = Image.fromarray(data)
            new_img.save(img_path)
            
        except Exception as e:
            print(f"Post-processing transparent failed: {e}")

def main():
    pkl_path = "/home/heliulu/ljn/EmbodiedScan/data/embodiedscan_infos_val.pkl" 
    data_root = "/home/heliulu/ljn/EmbodiedScan/data/matterport3d/" # "/home/heliulu/ljn/EmbodiedScan/data/scannet/scans/"
    output_dir = "/home/heliulu/ljn/EmbodiedScan/vis_matterport3d_gt_transparent/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.exists(pkl_path):
        print(f"Error: Pickle file not found at {pkl_path}")
        return

    visualizer = OccGtVisualizerStrict(pkl_path, output_dir) 

    all_scans = sorted(os.listdir(data_root))
    print(f"Found {len(all_scans)} scans. Processing GT (Open3D + Transparent Post-process)...")

    for scan_id in tqdm(all_scans):
        # 1. 构造输入文件路径
        for i in range(13):
            idx = str(i)
            region = f"region{idx}"
            npy_file = os.path.join(data_root, scan_id, "occupancy", f"occupancy_siglip_pred_{region}.npy")
            if os.path.exists(npy_file):
                npy_file = npy_file.replace("occupancy_siglip_pred", "occupancy")
                visualizer.process_and_save(npy_file, scan_id, region)

        print(f"Done! Saved to {output_dir}")

if __name__ == "__main__":
    main()
