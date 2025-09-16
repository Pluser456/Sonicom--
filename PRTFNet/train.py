import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from PRTFNet import PRTFNet
from new_dataset import HRTFDataSet
from utils import split_dataset, train_one_epoch_2d, evaluate
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 设备配置
    current_model = "2D" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    # weightname = "best_model.pth"
    weightname = "nopretrain"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = False  # 是否使用差值HRTF数据
    pos_num_per_batch = 6  # 每个batch中包含的位置数量，建议设置为2562的约数，2562=2*3*7*61

    if current_model == "3D":
        weightdir = "PRTFNet/Wi_3D"
        ear_dir = "Ear_voxel_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        # model = threeDResnet().to(device)
        inputform = "voxel"
    elif current_model == "2D":
        weightdir = "PRTFNet/Wi_2D"
        ear_dir = "Ear_image_gray_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        model = PRTFNet().to(device)
        inputform = "image"
    # 计算总参数量
    print(f"{'Layer Name':<60} | {'Parameters':>15}")
    print("-" * 80)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        
        params = parameter.numel() # 获取参数的数量
        print(f"{name:<60} | {params:>15,}")
        total_params += params

    print("-" * 80)
    print(f"{'Total Trainable Params':<60} | {total_params:>15,}")
    print(f"Total Trainable Params (M): {total_params / 1e6:.2f}M")

    modelpath = f"{weightdir}/{weightname}"
    if os.path.exists(modelpath):
        print("Load model from", modelpath)
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    
    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)

    # 创建数据集
    train_dataset = HRTFDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=usediff,
        calc_mean=usediff,
        inputform=inputform,
        mode="right",
        pos_num_per_batch=pos_num_per_batch
    )
    
    test_dataset = HRTFDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=inputform,
        mode="right",
        use_diff=usediff,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right,
        pos_num_per_batch=pos_num_per_batch
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # 请固定此batch_size为1，或调整上面的pos_num_per_batch
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1, # 请固定此batch_size为1，或调整上面的pos_num_per_batch
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    writer = SummaryWriter(log_dir=f"{weightdir}/test_{time.strftime('%m-%d_%H-%M')}")
    # 训练循环
    num_epochs = 20
    best_loss = 300
    
    patience = 5  # 早停的容忍次数
    patience_counter = 0

    for epoch in range(0, num_epochs + 1):
        train_dataset.on_epoch_end() # 打乱训练集的数据顺序
        test_dataset.on_epoch_end() # 打乱测试集的数据顺序

        # 训练
        loss = train_one_epoch_2d(model, optimizer, train_loader, device, epoch)

        # 验证
        val_loss = evaluate(model, test_loader, device, epoch)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        scheduler.step()
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        # 检查是否是最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), f"{weightdir}/best_model.pth")
            print(f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1

        # 检查早停条件
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs with best validation loss: {best_loss:.4f}")
            break

        # 保存当前模型
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), f"{weightdir}/model-{epoch}.pth")
        #     print(f"Saved model at epoch {epoch}")

if __name__ == "__main__":
    main()