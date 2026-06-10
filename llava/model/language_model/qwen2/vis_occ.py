import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image  # 需要 pip install pillow
from tqdm import tqdm  # 需要 pip install tqdm

class OccPredDrawerHeadless:
    def __init__(self, npy_path, save_path, voxel_size=0.16):
        self.npy_path = npy_path
        self.save_path = save_path
        self.voxel_size = voxel_size
        
        # 定义点云范围 (Scannet/Matterport3d 常用范围)
        # 格式: [x_min, y_min, z_min, x_max, y_max, z_max]
        self.point_cloud_range = np.array([-3.2, -3.2, -1.28 + 0.5, 3.2, 3.2, 1.28 + 0.5])
        
        # 移植 Mayavi 风格的颜色表 (RGB 0-1)
        self.colors_palette = np.array([
            [255, 120,  50, 255], [255, 192, 203, 255], [255, 255,   0, 255],
            [  0, 150, 245, 255], [  0, 255, 255, 255], [0, 175,   0, 255],
            [255,   0,   0, 255], [127, 127, 127, 255], [135,  60,   0, 255],
            [160,  32, 240, 255],
        ]).astype(np.float64) / 255.0
        self.colors_palette = self.colors_palette[:, :3] # 去掉 Alpha 通道

    def run(self):
        if not os.path.exists(self.npy_path):
            return

        # 1. 加载数据
        # 假设数据格式为 (N, 4) -> [x_grid, y_grid, z_grid, label_id]
        data = np.load(self.npy_path)
        
        if data.shape[0] == 0:
            print(f"Skipping {os.path.basename(self.save_path)}: Empty data.")
            return

        # 2. 数据预处理 (向量化加速版本)
        # 提取 label
        labels = data[:, 3].astype(int)
        
        # 过滤掉 label <= 0 的点
        valid_mask = labels > 0
        data = data[valid_mask]
        labels = labels[valid_mask]

        z_height_mask = data[:, 2] <= 12
        
        data = data[z_height_mask]
        labels = labels[z_height_mask]

        # --- 颜色过滤逻辑 (严格模式) ---
        # 你的要求：如果 class id 超过颜色数量，就不可视化
        # labels 是 1-based, 索引是 label-1
        color_indices = labels - 1
        color_valid_mask = color_indices < len(self.colors_palette)
        
        # 应用颜色过滤
        data = data[color_valid_mask]
        color_indices = color_indices[color_valid_mask]
        
        if len(data) == 0:
            print(f"Skipping {os.path.basename(self.save_path)}: No valid classes within color palette range.")
            return

        # 3. 坐标转换 (Grid Index -> World Coordinate)
        # 向量化计算： world = grid * voxel_size + min_range + half_voxel
        grid_coords = data[:, :3]
        min_range = self.point_cloud_range[:3]
        
        points = grid_coords * self.voxel_size + min_range + (self.voxel_size / 2)
        
        # 4. 获取颜色
        point_colors = self.colors_palette[color_indices]

        # 5. 创建 Open3D 几何体
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(point_colors)
        
        # 创建 VoxelGrid (用于更好的体素化显示)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=self.voxel_size
        )
        
        # 6. 渲染
        try:
            success = self._render_with_open3d_offscreen(voxel_grid)
            if success:
                self._make_transparent(self.save_path)
        except Exception as e:
            # print(f"Open3D failed for {self.save_path}: {e}. Trying Matplotlib...")
            self._render_with_matplotlib(points, point_colors)

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

    def _render_with_open3d_offscreen(self, geometry):
        width, height = 1920, 1080
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
        
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("voxels", geometry, mat)
        
        # --- 调整相机视角 ---
        center = geometry.get_center()
        # 之前的 offset = [0, 6, 6] 可能太近或者角度比较单一
        # 这里改为 ISO 视角 (45度鸟瞰)，能看清更多细节
        offset = [6, -6, 6] 
        eye = center + offset
        
        renderer.setup_camera(60.0, center, eye, [0, 0, 1])
        
        img = renderer.render_to_image()
        o3d.io.write_image(self.save_path, img)
        return True

    def _render_with_matplotlib(self, points, colors):
        """备用方案"""
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 降采样以防卡死
        if len(points) > 5000:
            indices = np.random.choice(len(points), 5000, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='s', s=10)
        ax.view_init(elev=30, azim=45) 
        plt.savefig(self.save_path)
        plt.close()

def main():
    # --- 配置路径 ---
    data_root = "/home/heliulu/ljn/EmbodiedScan/data/3rscan/" # "/home/heliulu/ljn/EmbodiedScan/data/scannet/scans/" # "/home/heliulu/ljn/EmbodiedScan/data/scannet"
    output_dir = "/home/heliulu/ljn/EmbodiedScan/vis_3rscan_cylinder_testset_transparent/"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有 scan 文件夹
    if not os.path.exists(data_root):
        print(f"Path not found: {data_root}")
        return

    all_scans = sorted(os.listdir(data_root))
    print(f"Found {len(all_scans)} potential scans. Processing...")

    # 使用 tqdm 进度条遍历
    for scan_id in tqdm(all_scans):
        # 1. 构造输入文件路径
        # for i in range(13):
        #     idx = str(i)
        #     region = f"region{idx}"
        npy_file = os.path.join(data_root, scan_id, "occupancy", "occupancy_cylinder_testset_pred_defalut.npy")
        
        # 2. 检查文件是否存在
        if os.path.exists(npy_file):
            # 3. 构造输出文件路径 (保留 scan_id)
            save_name = f"{scan_id}_vis.png"
            save_file = os.path.join(output_dir, save_name)
            
            # 可选：如果文件已存在跳过
            # if os.path.exists(save_file): continue
            # 4. 执行可视化 
            drawer = OccPredDrawerHeadless(npy_file, save_path=save_file)
            drawer.run()

    print(f"Done! Results saved to {output_dir}")

if __name__ == "__main__":
    main()
