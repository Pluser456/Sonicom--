import torch

# 指定模型保存的路径
checkpoint_path = "NewVAECode/checkpoints/vae_incept_edges.ckpt"

# 加载模型的状态字典
#checkpoint = torch.load(checkpoint_path)
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# 打印状态字典的内容
print(checkpoint.keys())

# 获取 state_dict
state_dict = checkpoint['state_dict']

# 遍历 state_dict 并打印每一层的大小
#for key, value in state_dict.items():
#    print(f"Layer: {key}, Size: {value.size()}")
