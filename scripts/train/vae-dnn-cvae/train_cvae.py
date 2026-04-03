"""
CVAE 训练脚本 - 支持配置文件
用法:
    python train_cvae.py --config configs/vae-dnn-cvae/cvae_default.yaml
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import sys

from src.utils.config import load_config
from src.dataset.hybrid import CVAEDataSet
from src.utils.data import split_dataset
from src.models.hybrid.cvae import CVAE
from src.utils.training import create_experiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CVAE Training')
    parser.add_argument('--config', type=str, default='configs/vae-dnn-cvae/cvae_default.yaml',
                        help='Path to config file')
    parser.add_argument('--weightname', type=str, default='nopretrain',
                        help='Weight file name')
    return parser.parse_args()


def loss_function(hrtf_true, hrtf_pred, means, log_var):
    """
    CVAE 损失函数
    Args:
        hrtf_true: 真实 HRTF 频响 [batch_size, nfft]
        hrtf_pred: 重构 HRTF 频响 [batch_size, nfft]
        means: 潜在空间均值 [batch_size, latent_size]
        log_var: 潜在空间对数方差 [batch_size, latent_size]
    Returns:
        mse: 重构损失
        kld: KL 散度
        loss: 总损失
    """
    mse = nn.functional.mse_loss(hrtf_pred, hrtf_true, reduction='sum') / hrtf_true.size(0)
    kld = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp()) / hrtf_true.size(0)
    loss = mse + kld
    return mse, kld, loss


def main():
    args = parse_args()

    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        raise FileNotFoundError(f"Config file {args.config} not found.")

    # 设置设备和路径
    device = torch.device(config.training.device)
    weightdir = config.paths.checkpoint_dir
    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)
    log_dir = config.paths.log_dir

    if not os.path.exists(weightdir):
        os.makedirs(weightdir)

    modelpath = f"{weightdir}/{args.weightname}"

    # 加载数据集
    dataset_paths = split_dataset(ear_dir, hrtf_dir, inputform=config.dataset.input_form,
                                  n_folds=config.dataset.n_folds, val_fold=config.dataset.val_fold,
                                  seed=config.dataset.seed)

    # 创建数据集
    train_dataset = CVAEDataSet(
        dataset_paths["train_hrtf_list"],
        use_diff=config.dataset.use_diff,
        calc_mean=config.dataset.use_diff,
        status="train",
        mode=config.dataset.mode
    )
    test_dataset = CVAEDataSet(
        dataset_paths["test_hrtf_list"],
        use_diff=config.dataset.use_diff,
        calc_mean=False,
        status="test",
        mode=config.dataset.mode,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # 构建完整的网络层结构（与 CVAECfg 逻辑一致）
    nfft = config.model.nfft
    encoder_layer_sizes = [nfft] + config.model.encoder_layer_sizes  # [nfft, ...中间层]
    decoder_layer_sizes = config.model.decoder_layer_sizes + [nfft]  # [...中间层, nfft]
    print(f"Encoder layers: {encoder_layer_sizes}")
    print(f"Decoder layers: {decoder_layer_sizes}")

    # 模型实例化
    model = CVAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=config.model.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        num_labels=config.model.num_labels
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    num_epochs = config.training.epochs

    # 使用 ReduceLROnPlateau 调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        cooldown=config.training.scheduler_cooldown
    )

    # 加载已有权重
    start_epoch = 0
    if config.training.continue_exp is not None:
        if os.path.exists(modelpath):
            checkpoint = torch.load(modelpath, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Load model from {modelpath}")
        else:
            raise FileNotFoundError(f"Checkpoint {modelpath} not found for continuation")

    # 创建实验文件夹并保存配置
    writer = None
    if config.training.log:
        experiment_id, writer = create_experiment(log_dir, config, start_epoch)

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss_recon = 0
        epoch_loss_kl = 0

        train_progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            file=sys.stdout
        )

        for i, batch in enumerate(train_progress_bar):
            # 获取 HRTF 数据和条件标签
            hrtf_true = batch["hrtf"].to(device)
            condition = batch["position"].float().to(device)  # [batch, 2]: [az, el]

            optimizer.zero_grad()

            # 模型前向
            hrtf_pred, mu, logvar, z = model(hrtf_true, condition)
            recon_loss, kl_loss, total_loss = loss_function(hrtf_true, hrtf_pred, mu, logvar)

            total_loss.backward()
            optimizer.step()

            epoch_loss_recon += recon_loss.item()
            epoch_loss_kl += kl_loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            train_progress_bar.set_description(
                f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.4f} "
                f"loss_kl: {epoch_loss_kl/(i+1):.4f} "
                f"lr: {current_lr:.2e}"
            )

        avg_recon_loss_train = epoch_loss_recon / len(train_loader)
        avg_kl_loss_train = epoch_loss_kl / len(train_loader)

        if writer is not None:
            writer.add_scalar("train_loss_recon", avg_recon_loss_train, epoch)
            writer.add_scalar("train_loss_kl", avg_kl_loss_train, epoch)
            writer.add_scalar("train_loss_total", avg_recon_loss_train + avg_kl_loss_train, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # 验证
        model.eval()
        val_loss_recon = 0
        val_loss_kl = 0

        with torch.no_grad():
            val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
            for step, batch in enumerate(val_progress_bar):
                hrtf_val = batch["hrtf"].to(device)
                condition_val = batch["position"].float().to(device)  # [batch, 2]: [az, el]

                hrtf_pred_val, mu_val, logvar_val, _ = model(hrtf_val, condition_val)
                recon_loss_val, kl_loss_val, _ = loss_function(hrtf_val, hrtf_pred_val, mu_val, logvar_val)

                val_loss_recon += recon_loss_val.item()
                val_loss_kl += kl_loss_val.item()

                val_progress_bar.desc = (
                    f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.4f} "
                    f"loss_kl: {val_loss_kl/(step+1):.4f}"
                )

        avg_recon_loss_val = val_loss_recon / len(test_loader)
        avg_kl_loss_val = val_loss_kl / len(test_loader)

        if writer is not None:
            writer.add_scalar("val_loss_recon", avg_recon_loss_val, epoch)
            writer.add_scalar("val_loss_kl", avg_kl_loss_val, epoch)
            writer.add_scalar("val_loss_total", avg_recon_loss_val + avg_kl_loss_val, epoch)

        # 更新学习率
        scheduler.step(avg_recon_loss_train + avg_kl_loss_train)

        print("")

        # 保存模型
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None
            }
            torch.save(checkpoint, f"{weightdir}/exp_{experiment_id:03d}_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}")

    if writer is not None:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
