import os
import h5py
import numpy as np

image_dir = "Ear_image_gray"
hrtf_dir = "FFT_HRTF"
test_image_dir = "testdataimage"
test_hrtf_dir = "test_hubuts_windowed_HRTF"

cwd = os.path.dirname(os.path.realpath(__file__))

parent_dir = os.path.dirname(cwd)
image_dir = os.path.join(parent_dir, image_dir)
hrtf_dir = os.path.join(parent_dir, hrtf_dir)

testidxs = [7, 14, 27, 30, 31, 52, 54, 55, 70, 82, 143, 184]
# print(sorted(np.random.choice(190, size=12, replace=False)))

image_list =  os.listdir(image_dir)
image_list.sort(key=lambda x:int(x.split('.')[0].split('_')[0][1:]))  # 按P后面的数字排序

# 分离左右耳图像
left_image_list = [img for img in image_list if img.endswith('left.png')]
right_image_list = [img for img in image_list if img.endswith('right.png')]

left_train = [x for i, x in enumerate(left_image_list) if i not in testidxs]
right_train = [x for i, x in enumerate(right_image_list) if i not in testidxs]    
left_test = [x for i, x in enumerate(left_image_list) if i in testidxs]
right_test = [x for i, x in enumerate(right_image_list) if i in testidxs]

train_people_num = len(left_train)
test_people_num = len(left_test)

# 从left_image_list中提取编号
train_image_numbers = [int(img.split('_')[0][1:]) for img in left_train]
test_image_numbers = [int(img.split('_')[0][1:]) for img in left_test]

# 过滤hrtf_list，只保留编号在image_numbers中的文件
train_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in train_image_numbers]
train_hrtf_list = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in train_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]
test_hrtf_list = [x for x in os.listdir(hrtf_dir) if int(x.split('.')[0][1:]) in test_image_numbers]
test_hrtf_list = [os.path.join(hrtf_dir, hrtf_file_name) for hrtf_file_name in test_hrtf_list if os.path.isfile(os.path.join(hrtf_dir, hrtf_file_name))]

# 获取图像完整路径
left_train = [os.path.join(image_dir, img) for img in left_train]
right_train = [os.path.join(image_dir, img) for img in right_train]
left_test = [os.path.join(image_dir, img) for img in left_test]
right_test = [os.path.join(image_dir, img) for img in right_test]

# model related
model_path = "model.pth"
model_path = os.path.join(cwd, model_path)

def calculate_hrtf_mean(hrtf_file_names, whichear=None):
    # 初始化累加器和计数器
    hrtf_sum = None
    total_samples = 0
    
    for file_path in hrtf_file_names:
        with h5py.File(file_path, 'r') as data:
            # 读取当前文件所有位置的HRTF数据
            hrtfs = data[f'F_{whichear}'][:]  # 形状为 (num_positions, num_freq_bins)
            
            # 如果是第一次读取，初始化累加器
            if hrtf_sum is None:
                hrtf_sum = np.zeros(hrtfs.shape, dtype=np.float64)
            
            # 累加当前文件所有位置的HRTF
            hrtf_sum += hrtfs
            total_samples += 1
    
    # 计算全局平均
    hrtf_mean = hrtf_sum / total_samples
    return hrtf_mean  # 保持与原始数据相同的精度
