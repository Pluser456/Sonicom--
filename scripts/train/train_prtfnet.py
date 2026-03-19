"""
PRTFNet 训练脚本 - 支持配置文件
用法:
    python Train_prtfnet.py --config configs/default.yaml
"""
import os
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.prtfnet import PRTFNet
from src.dataset.prtf import HRTFDataSet
from src.utils.data import split_dataset
from src.utils.config import load_config, get_default_config
from src.utils.training import create_experiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PRTFNet Training')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--weightname', type=str, default='nopretrain',
                        help='Weight file name')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file {args.config} not found, using defaults")
        config = get_default_config()

    # 设置设备和路径
    device = torch.device(config.training.device)
    weightdir = config.paths.checkpoint_dir
    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)
    log_dir = config.paths.log_dir

    if not os.path.exists(weightdir):
        os.makedirs(weightdir)

    modelpath = f"{weightdir}/{args.weightname}"

    # 数据分割
    dataset_paths = split_dataset(
        ear_dir, hrtf_dir,
        inputform=config.dataset.input_form,
        n_folds=config.dataset.n_folds,
        val_fold=config.dataset.val_fold,
        seed=config.dataset.seed
    )

    # 创建数据集
    train_dataset = HRTFDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=config.dataset.use_diff,
        calc_mean=config.dataset.use_diff,
        inputform=config.dataset.input_form,
        mode=config.dataset.mode,
        pos_num_per_batch=config.dataset.pos_num_per_batch
    )

    test_dataset = HRTFDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=config.dataset.input_form,
        mode=config.dataset.mode,
        use_diff=config.dataset.use_diff,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right,
        pos_num_per_batch=config.dataset.pos_num_per_batch
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # 模型实例化
    model = PRTFNet(pos_num=config.dataset.pos_num, freq_num=config.dataset.freq_num).to(device)

    # 计算总参数量
    print(f"{'Layer Name':<60} | {'Parameters':>15}")
    print("-" * 80)

    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        params = parameter.numel()
        print(f"{name:<60} | {params:>15,}")
        total_params += params

    print("-" * 80)
    print(f"{'Total Trainable Params':<60} | {total_params:>15,}")
    print(f"Total Trainable Params (M): {total_params / 1e6:.2f}M")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    num_epochs = config.training.epochs
    total_steps = num_epochs * len(train_loader)  # 总训练步数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.training.learning_rate * 0.01)
    loss_function = nn.MSELoss()

    # 加载已有权重
    start_epoch = 0
    if config.training.continue_exp is not None:
        if os.path.exists(modelpath):
            checkpoint = torch.load(modelpath, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Load model from {modelpath}")
        else:
            raise FileNotFoundError(f"Checkpoint {modelpath} not found for continuation")
    else:
        print("Checkpoint provided but continue_exp is None, starting fresh training")

    # 创建实验文件夹并保存配置
    if config.training.log == True:
        experiment_id, writer = create_experiment(log_dir, config, start_epoch)

    # 训练相关参数
    update_interval = config.training.scheduler_update_interval  # 每 462 步更新一次 scheduler
    global_step = start_epoch * len(train_loader)  # 全局步数，用于 writer 记录
    loss_window = []  # 用于记录最近 462 步的 loss

    for epoch in range(start_epoch, num_epochs):
        train_dataset.on_epoch_end()
        test_dataset.on_epoch_end()

        # ===== 训练 =====
        model.train()
        accu_loss = torch.zeros(1).to(device)

        optimizer.zero_grad()

        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]", file=sys.stdout)

        for step, sample_batch in enumerate(train_progress_bar):
            one_hot = sample_batch["one_hot"].type(torch.float32)
            hrtf = sample_batch["hrtf"].to(device)
            right_voxel = sample_batch["right_voxel"]

            optimizer.zero_grad()

            mu = model(right_voxel, one_hot, device=device)
            loss = loss_function(mu, hrtf)

            loss.backward()
            accu_loss += loss.detach()
            loss_window.append(loss.detach().item())
            optimizer.step()
            scheduler.step()

            # 保持窗口大小为 update_interval
            if len(loss_window) > update_interval:
                loss_window.pop(0)

            # 计算最近 462 步的平均 loss
            avg_loss = sum(loss_window) / len(loss_window)
            train_progress_bar.desc = (
                f"[train epoch {epoch+1}] loss: {avg_loss:.3f} "
                f"lr: {optimizer.param_groups[0]['lr']:.4e}"
            )

            # 每 update_interval 步进行日志记录
            if (global_step + step + 1) % update_interval == 0:
                if config.training.log == True:
                    writer.add_scalar("Loss/train_step", avg_loss, global_step + step + 1)
                    writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step + step + 1)

        # 处理剩余步数（不足 462 步的部分）
        if (global_step + step + 1) % update_interval != 0:
            avg_loss = sum(loss_window) / update_interval
            if config.training.log == True:
                writer.add_scalar("Loss/train_step", avg_loss, global_step + step + 1)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step + step + 1)

        # 更新全局步数
        global_step += len(train_loader)

        # 计算整个 epoch 的平均 loss（用于记录）
        loss = accu_loss.item() / len(train_loader)

        # ===== 验证 =====
        model.eval()
        accu_val_loss = torch.zeros(1).to(device)

        with torch.no_grad():
            val_progress_bar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [valid]", file=sys.stdout)
            for step, sample_batch in enumerate(val_progress_bar):
                one_hot = sample_batch["one_hot"].type(torch.float32)
                hrtf = sample_batch["hrtf"].to(device)
                right_voxel = sample_batch["right_voxel"]

                mu = model(right_voxel, one_hot, device=device)
                val_loss = loss_function(mu, hrtf)

                accu_val_loss += val_loss.detach()
                val_progress_bar.desc = (
                    f"[valid epoch {epoch+1}] loss: {accu_val_loss.item()/(step+1):.3f}"
                )

        val_loss = accu_val_loss.item() / len(test_loader)

        if config.training.log == True:
            writer.add_scalar("Loss/train_epoch", loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)

        print("\n")

        # 每个 epoch 保存一次模型权重
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, f"{weightdir}/exp_{experiment_id:03d}_epoch_{epoch+1}.pt")
        print(f"Checkpoint saved at epoch {epoch+1}")
        
    print("Training finished.")


if __name__ == "__main__":
    main()