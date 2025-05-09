import os

# 定义两个文件夹的路径
ear_image_path = "Ear_image_gray"
ear_voxel_path = "Ear_voxel"

def get_file_prefixes(directory):
    """获取目录中所有文件的前缀（文件名去掉扩展名）"""
    prefixes = set()
    for filename in os.listdir(directory):
        prefix = os.path.splitext(filename)[0]
        prefixes.add(prefix)
    return prefixes

# 获取两个文件夹中的文件前缀
image_prefixes = get_file_prefixes(ear_image_path)
voxel_prefixes = get_file_prefixes(ear_voxel_path)

# 找出只在image文件夹中的前缀
only_in_image = image_prefixes - voxel_prefixes
if only_in_image:
    print("只在Ear_image_gray中存在的前缀：")
    for prefix in sorted(only_in_image):
        print(f"- {prefix}")

# 找出只在voxel文件夹中的前缀
only_in_voxel = voxel_prefixes - image_prefixes
if only_in_voxel:
    print("\n只在Ear_voxel中存在的前缀：")
    for prefix in sorted(only_in_voxel):
        print(f"- {prefix}")

# 如果都完全相同，输出提示
if not only_in_image and not only_in_voxel:
    print("两个文件夹中的文件前缀完全相同。")