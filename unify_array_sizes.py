import numpy as np
import os

def unify_npy_array_sizes(directory):
    """
    统一目录下所有 .npy 文件的数组尺寸。

    1. 遍历目录，找到所有 .npy 文件中数组的最大高度、宽度和深度。
    2. 再次遍历，将尺寸不足的数组用 0 填充至最大尺寸，并覆盖保存。

    Args:
        directory (str): 包含 .npy 文件的目录路径。
    """
    max_h, max_w, max_d = 0, 0, 0
    npy_files = []

    print(f"开始扫描目录: {directory}")

    # 第一次循环：查找最大尺寸
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            filepath = os.path.join(directory, filename)
            npy_files.append(filepath)
            try:
                arr = np.load(filepath)
                # 假设数组是 3D 的 (height, width, depth)
                if arr.ndim == 3:
                    h, w, d = arr.shape
                    max_h = max(max_h, h)
                    max_w = max(max_w, w)
                    max_d = max(max_d, d)
                else:
                    print(f"警告: 文件 {filename} 不是 3D 数组，维度为 {arr.ndim}。已跳过。")
            except Exception as e:
                print(f"错误: 无法加载或处理文件 {filename}: {e}")

    if not npy_files:
        print("错误: 在指定目录下未找到 .npy 文件。")
        return

    print(f"扫描完成。找到的最大尺寸 (H, W, D): ({max_h}, {max_w}, {max_d})")

    # 第二次循环：填充并保存
    print("开始统一数组尺寸...")
    for filepath in npy_files:
        filename = os.path.basename(filepath)
        try:
            arr = np.load(filepath)
            if arr.ndim != 3:
                # 跳过非 3D 数组，因为填充逻辑是基于 3D 的
                continue

            h, w, d = arr.shape
            # 检查是否需要填充
            if h < max_h or w < max_w or d < max_d:
                pad_h = max_h - h
                pad_w = max_w - w
                pad_d = max_d - d

                # 定义填充宽度，(before, after) 对每个维度
                # 我们只在数组的末尾填充
                pad_width = ((0, pad_h), (0, pad_w), (0, pad_d))

                # 使用 0 进行填充
                padded_arr = np.pad(arr, pad_width, mode='constant', constant_values=0)

                # 覆盖保存
                np.save(filepath, padded_arr)
                print(f"已填充并保存: {filename} (原尺寸: ({h},{w},{d}), 新尺寸: ({max_h},{max_w},{max_d}))")
            # else:
            #     print(f"尺寸已满足，无需填充: {filename}")

        except Exception as e:
            print(f"错误: 处理或保存文件 {filename} 时出错: {e}")

    print("所有 .npy 文件尺寸统一完成。")

# --- 使用示例 ---
# !!! 重要: 请将下面的 'your_npy_folder_path' 替换为实际包含 .npy 文件的文件夹路径 !!!
target_directory = r'D:\大创相关\Sonicom网络\Ear_voxel' # 例如: r'D:\data\my_npy_files'

if os.path.isdir(target_directory):
    unify_npy_array_sizes(target_directory)
else:
    print(f"错误: 目录 '{target_directory}' 不存在或不是一个有效的目录。")
