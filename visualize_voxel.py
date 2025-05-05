import open3d as o3d
import numpy as np
import os

def numpy_to_voxel_grid(voxel_array, voxel_size=1.0, origin=[0,0,0]):
    """将 3D NumPy 数组转换为 Open3D VoxelGrid (通过点云转换)"""
    if voxel_array is None or voxel_array.ndim != 3:
        print("错误：输入的 NumPy 数组无效。")
        return None

    # 找到所有被占用的体素的索引 (值为 1 的位置)
    indices = np.argwhere(voxel_array == 1)
    if indices.shape[0] == 0:
        print("警告：NumPy 数组中没有被占用的体素。")
        # 返回一个空的 VoxelGrid 对象
        voxel_grid = o3d.geometry.VoxelGrid()
        voxel_grid.voxel_size = voxel_size
        voxel_grid.origin = np.array(origin)
        return voxel_grid

    # --- 通过点云转换 ---
    # 1. 将体素索引转换为点坐标 (体素中心)
    #    假设 NumPy 数组索引顺序是 x, y, z
    points = (indices + 0.5) * voxel_size + np.array(origin)

    # 2. 创建一个临时点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5]) # 可选

    # 3. 从点云创建 VoxelGrid
    try:
        # 使用点云的边界来确定原点，可能更准确
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    except Exception as e:
        print(f"从点云创建 VoxelGrid 时出错: {e}")
        return None
    # --- 转换结束 ---

    num_created_voxels = len(voxel_grid.get_voxels())
    print(f"成功从 NumPy 数组创建 VoxelGrid (通过点云转换)，包含 {num_created_voxels} 个体素。")
    return voxel_grid


def visualize_with_custom_view(voxel_grid, filename):
    """使用自定义视角可视化 VoxelGrid"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Voxel Visualization - {filename}", width=1600, height=1200)
    vis.add_geometry(voxel_grid)

        # --- 添加坐标轴 ---
    # 计算坐标轴的大小，可以基于体素网格的范围
    aabb = voxel_grid.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    max_extent = max(extent) if any(extent) else 50 # 如果范围为0，给一个默认大小
    axis_size = max_extent * 0.3 # 坐标轴大小设为最大范围的30%，可调整

    # 创建坐标轴网格 (X: 红, Y: 绿, Z: 蓝)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=axis_size,
        origin=[0, 0, 0] # 将坐标轴放在原点
        # origin=aabb.get_min_bound() # 或者放在包围盒的最小角点
    )
    vis.add_geometry(mesh_frame) # 将坐标轴添加到可视化器
    # 获取视图控制器
    ctr = vis.get_view_control()

    # 计算相机参数
    center = voxel_grid.get_center()
    aabb = voxel_grid.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    max_extent = max(extent) if any(extent) else 100 # 防止 extent 全为0
    
    # 默认向上向量
    up = [0, 0, 1] # 假设Z轴是向上的
    
    # 根据文件名判断耳朵朝向并设置相机位置 (eye)
    distance_factor = 1.5 # 相机距离中心的系数，可调整
    eye_distance = max_extent * distance_factor

    # 默认视角：从前方稍偏上观察
    eye = center + np.array([0, -eye_distance, eye_distance * 0.3]) # 默认从Y负方向看

    if "_left" in filename.lower():
        print("  检测到左耳，设置相机从左侧观察...")
        # 相机放在 Y 轴正方向
        eye = center + np.array([0, -eye_distance, eye_distance * 0.2]) # Y负方向，稍微偏上
    elif "_right" in filename.lower():
        print("  检测到右耳，设置相机从右侧观察...")
        # 相机放在 Y 轴负方向
        eye = center + np.array([0, eye_distance, eye_distance * 0.2]) # Y正方向，稍微偏上
    else:
        print("  未检测到左右耳标识，使用默认前方视角...")
        # 可以设置一个默认的前方视角，例如从X正方向
        # eye = center + np.array([eye_distance, 0, eye_distance * 0.2])
    # eye = center + np.array([0, -eye_distance, eye_distance * 0.2]) # Y负方向，稍微偏上

    # 设置相机参数
    # 注意：set_front 的参数是相机朝向的向量 (lookat - eye)
    # set_lookat 设置焦点
    # set_up 设置哪个方向是上方
    ctr.set_up(up)
    ctr.set_lookat(center)
    ctr.set_front(center - eye) # 向量：从 eye 指向 center
    # ctr.set_zoom(0.8) # 可选：调整缩放级别

    # 运行可视化 (阻塞，直到窗口关闭)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    VOXEL_DIR = "Ear_voxel"
    VOXEL_SIZE_FOR_VIS = 0.582 # 使用与创建时相同或合适的体素大小进行可视化

    if not os.path.isdir(VOXEL_DIR):
        print(f"错误：文件夹 '{VOXEL_DIR}' 不存在。")
        exit()

    print(f"开始可视化 '{VOXEL_DIR}' 中的体素数据...")

    npy_files = sorted([f for f in os.listdir(VOXEL_DIR) if f.endswith('.npy')])

    if not npy_files:
        print(f"文件夹 '{VOXEL_DIR}' 中没有找到 .npy 文件。")
        exit()
    start_idx = 165
    start_idx =2*start_idx
    npy_files = npy_files[start_idx:] # 只可视化第51个到最后一个文件
    for filename in npy_files:
        file_path = os.path.join(VOXEL_DIR, filename)
        print(f"\n--- 正在加载和可视化: {filename} ---")

        try:
            # 1. 加载 NumPy 数组
            voxel_array = np.load(file_path)
            if "_right" in filename.lower():
                voxel_array = np.flip(voxel_array, axis=1)

            print(f"  数组形状: {voxel_array.shape}")
            print(f"  占用体素数: {np.sum(voxel_array)}")

            # 2. 将 NumPy 数组转换为 VoxelGrid
            voxel_grid = numpy_to_voxel_grid(voxel_array, voxel_size=VOXEL_SIZE_FOR_VIS, origin=[0,0,0]) # origin 可能需调整

            if voxel_grid and len(voxel_grid.get_voxels()) > 0:
                # 3. 使用自定义视角可视化 VoxelGrid
                print("  显示体素网格... 关闭窗口以继续下一个。")
                visualize_with_custom_view(voxel_grid, filename) # 调用新的可视化函数
            elif voxel_grid:
                 print("  生成的 VoxelGrid 为空，跳过可视化。")
            else:
                print("  无法创建 VoxelGrid 进行可视化。")

        except Exception as e:
            print(f"  处理文件 {filename} 时出错: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- 可视化完成 ---")