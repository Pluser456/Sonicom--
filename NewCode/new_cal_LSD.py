import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from utils import *
from vit_model import vit_base_patch16_224_in21k as create_model
import matplotlib.pyplot as plt
import os
import argparse
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet, SingleSubjectDataSet

# from utils import read_split_data, train_one_epoch, evaluate

model_path = "D:\大学\大三下\大创项目\新数据库\Sonicom--\weights\model-0.pth"

def evaluate_one_hrtf(args, model, test_loader):
    model.eval()

    all_preds = []
    all_targets = []


    with torch.no_grad():
        for batch in test_loader:
            # 数据迁移到设备
            imageleft = batch["imageleft"].to(device)
            imageright = batch["imageright"].to(device)
            position = batch["position"].to(device)
            hrtf = batch["hrtf"].to(device)  # [batch]
            meanloghrtf = batch["meanlog"].to(device)  # [batch]

            # 前向传播
            # outputs = model(imageleft,imageright, position)  # [batch]
            outputs = model(imageleft, position)  # [batch]
            targets = hrtf.squeeze(1)
            # 添加epsilon防止log(0)
            targets = targets + 1e-8

            # 转换到对数域 (dB)
            log_target = 20 * torch.log10(targets)
            pred = torch.abs(outputs + meanloghrtf)
            log_target = torch.abs(log_target)

            # 将当前batch的结果添加到列表
            all_preds.append(pred)
            all_targets.append(log_target)
            # print(outputs.shape)
            # print(meanloghrtf.shape)


    # 将所有batch的结果拼接成两个大矩阵
    final_preds = torch.cat(all_preds, dim=0)  # [total_samples, n_frequencies]
    final_targets = torch.cat(all_targets, dim=0)  # [total_samples, n_frequencies]

    return final_preds, final_targets




if __name__ == '__main__':
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)


    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符 jx_vit_base_patch16_224_in21k-e5005f0a.pth
    parser.add_argument('--weights', type=str, default='./jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    # 2. 模型和训练配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)  # 使用之前定义的网络结构
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print("weight_dict.state_dict().keys():{}".format(weights_dict.keys()))
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['head.weight', 'head.bias'] # 'pre_logits.fc.weight', 'pre_logits.fc.bias',
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))
    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    # 如果需要加载预训练模型
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("Load model from", model_path)
    res_list = []
    pred_list = []
    true_list = []

    image_dir = "Ear_image_gray"
    hrtf_dir = "FFT_HRTF"

    dataset_paths = split_dataset(image_dir, hrtf_dir)
    # 获取各个数据集
    train_hrtf_list = dataset_paths['train_hrtf_list']
    test_hrtf_list = dataset_paths['test_hrtf_list']
    left_train = dataset_paths['left_train']
    right_train = dataset_paths['right_train']
    left_test = dataset_paths['left_test']
    right_test = dataset_paths['right_test']
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),  # 直接缩放到 224x224（可能改变长宽比）
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        "val": transforms.Compose([  # 验证集可保持原逻辑
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    # 实例化训练数据集
    train_dataset = SonicomDataSet(hrtf_files=train_hrtf_list,
                            left_images=left_train,
                            right_images=right_train,
                            transform=data_transform["train"],
                            mode="left")

    # 实例化验证数据集
    log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
    log_mean_hrtf_right = train_dataset.log_mean_hrtf_right

    batch_size = args.batch_size


    for hrtfid in range(1, 13):  # 选择计算第几个HRTF的LSD
        val_dataset = SingleSubjectDataSet(hrtf_files=test_hrtf_list,
                                           left_images=left_test,
                                           right_images=right_test,
                                           transform=data_transform["val"],
                                           mode="left",
                                           train_log_mean_hrtf_left=log_mean_hrtf_left,
                                           train_log_mean_hrtf_right=log_mean_hrtf_right,
                                           subject_id = hrtfid
                                           )
        dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size*6,
                                             shuffle=False,
                                             # pin_memory=True,
                                             # num_workers= nw,
                                             # collate_fn=val_dataset.collate_fn
                                                 )
        pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(args, model, dataloader)
        pred_list.append(pred_log_hrtf)
        true_list.append(true_log_hrtf)
        lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
        res_list.append(lsd)
        print(f"LSD of HRTF {hrtfid}:", lsd)

    print(f"Mean LSD: {np.mean(res_list)}")
    pred_tensor = torch.stack(pred_list, dim=0)
    true_tensor = torch.stack(true_list, dim=0)

    freq_list = np.linspace(0, 107, 108)  # 获取频率列表
    # 存储每个频率点的平均LSD
    avg_lsd_per_freq = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        # 计算平均LSD
        LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()
        print(f"Avg LSD of freq point {freq_idx}:{avg_lsd_per_freq[freq_idx]}")

    # 绘制频率-LSD图
    plt.figure(figsize=(10, 6))
    plt.semilogx(freq_list, avg_lsd_per_freq, 'b-o')
    plt.title('Frequency vs LSD')
    plt.xlabel('Frequency')
    plt.ylabel('LSD (dB)')
    plt.grid(True, which="both", ls="--")
    plt.show()
