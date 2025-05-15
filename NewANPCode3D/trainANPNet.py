import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from TestNet import TestNet as threeDResnetANP
from TestNet import ResNet3D as threeDResnet
from TestNet import ResNet2D as twoDResnet
from new_dataset import SonicomDataSet
from utils import split_dataset, train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    # 设备配置
    current_model = "3DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    weightname = "mode.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = True  # 是否使用差值HRTF数据

    if current_model == "3DResNetANP":
        weightdir = "./ANP3Dweights"
        ear_dir = "Ear_voxel"
        isANP = True
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)

        # 从预训练模型加载权重
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793 # 训练集每个文件选择的方位数
        model = threeDResnetANP(target_num_anp=5, positions_num=positions_chosen_num).to(device)
        inputform ="voxel"
    elif current_model == "3DResNet":
        weightdir = "./CNN3Dweights"
        ear_dir = "Ear_voxel"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793
        model = threeDResnet().to(device)
        inputform = "voxel"
    elif current_model == "2DResNet":
        weightdir = "./CNNweights"
        ear_dir = "Ear_image_gray"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793
        model = twoDResnet().to(device)
        inputform = "image"


    if os.path.exists(modelpath):
        print("Load model from", modelpath)
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    
    # 数据分割
    dataset_paths = split_dataset(ear_dir, "WinFFT_HRTF",inputform=inputform)
    
    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        positions_chosen_num=positions_chosen_num,
        use_diff=usediff,
        calc_mean=True,
        inputform=inputform,
        mode="left"
    )
    
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=inputform,
        mode="left",
        use_diff=usediff,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    auxiliary_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
    writer = SummaryWriter(log_dir=f"runs/{current_model}/test_{time.strftime('%m-%d_%H-%M')}")
    # 训练循环
    num_epochs = 480*5
    best_loss = 300
    
    patience = 30  # 早停的容忍次数
    patience_counter = 0

    for epoch in range(0, num_epochs + 1):
        # 训练
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 验证
        train_dataset.turn_auxiliary_mode(True)
        val_loss = evaluate(model, test_loader, device, epoch, auxiliary_loader=auxiliary_loader)
        train_dataset.turn_auxiliary_mode(False)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        scheduler.step()
        writer.add_scalar("Learning Rate", scheduler.get_last_lr()[0], epoch)
        # 检查是否是最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # 重置早停计数器
            # torch.save(model.state_dict(), f"{weightdir}/best_model.pth")
            # print(f"Saved best model with validation loss: {best_loss:.4f}")
            visualize_hrtf(model, test_loader, device, save_path=f"{weightdir}/visualization.png")
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

def visualize_hrtf(model, test_loader, device, save_path, max_samples=16):
    model.eval()
    hrtf_true_list = []
    hrtf_pred_list = []
    with torch.no_grad():
        count = 0
        for batch in test_loader:
            left_voxel = batch["left_voxel"]
            right_voxel = batch["right_voxel"]
            pos = batch["position"]
            hrtf_true = batch["hrtf"]
            hrtf_pred, _ = model(left_voxel, right_voxel, pos, hrtf_true, device=device)
            # 去除多余维度
            hrtf_true = hrtf_true.squeeze()
            hrtf_pred = hrtf_pred.squeeze()
            # 若数据是三维（形如 (batch, num_rows, features)），取第一个样本第一行的数据
            if hrtf_true.ndim == 3:
                sample_true = hrtf_true[0, 0, :].cpu().numpy()
                sample_pred = hrtf_pred[0, 0, :].cpu().numpy()
            # 若数据是二维（形如 (num_rows, features)），同样取第一行
            elif hrtf_true.ndim == 2:
                sample_true = hrtf_true[0, :].cpu().numpy()
                sample_pred = hrtf_pred[0, :].cpu().numpy()
            else:
                continue
            hrtf_true_list.append(sample_true)
            hrtf_pred_list.append(sample_pred)
            count += 1
            if count >= max_samples:
                break

    if len(hrtf_true_list) == 0:
        print("未获取到有效样本用于可视化")
        return

    # 确定网格行列数（尽量构成正方形）
    n_samples = len(hrtf_true_list)
    n_cols = int(np.ceil(np.sqrt(n_samples)))
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    # 将 axs 展平便于遍历
    axs = np.array(axs).reshape(-1)
    for i in range(n_rows * n_cols):
        ax = axs[i]
        if i < n_samples:
            ax.plot(hrtf_true_list[i], label="True HRTF", linewidth=2)
            ax.plot(hrtf_pred_list[i], label="Predicted HRTF", linestyle="--", linewidth=2)
            ax.set_title(f"Sample {i+1}", fontsize=10)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8)
        else:
            ax.axis('off')

    fig.suptitle("HRTF 对比网格", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"可视化图已保存至 {save_path}")

if __name__ == "__main__":
    main()