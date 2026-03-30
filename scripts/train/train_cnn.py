"""
CNN 分类器训练脚本 - 使用 ResNet3DClassifier 或 ResNet2DClassifier
从耳朵图像/体素预测 VQVAE 编码索引

使用方法:
    python scripts/train/train_cnn.py --config configs/vqvae/3dcnn-sub.yaml
    python scripts/train/train_cnn.py --config configs/vqvae/cnn-sub.yaml
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

from src.utils.config import load_config, get_default_config
from src.dataset.hrtf import CNNDataSet
from src.utils.data import split_dataset
from src.models.AE import HRTF_VQVAE
from src.models.TestNet import ResNet3DClassifier, ResNet2DClassifier
from src.utils.training import create_experiment


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CNN Training')
    parser.add_argument('--config', type=str, default='configs/vqvae/cnn-sub.yaml',
                        help='Path to config file')
    parser.add_argument('--weightname', type=str, default='nopretrain',
                        help='Weight file name')
    return parser.parse_args()


def load_pretrained_vqvae(config):
    """加载预训练的 VQVAE 模型"""
    device = torch.device(config.training.device)

    # 加载 VQVAE 模型配置
    vqvae_config_path = config.training.pretrained.vqvae_config
    vqvae_config = load_config(vqvae_config_path)

    model_config = vqvae_config.model

    # 创建模型
    vqvae_model = HRTF_VQVAE(
        hrtf_row_len=model_config.hrtf_row_len,
        encoder_out_vec_num=model_config.encoder_out_vec_num,
        embed_dim=model_config.embed_dim,
        encoder_transformer_config=model_config.transformer_encoder_settings,
        decoder_transformer_config=model_config.transformer_decoder_settings,
        num_embeddings=model_config.codebook_size,
        use_VQ=model_config.use_VQ,
        input_pos_as_seq=model_config.input_pos_as_seq,
        decay=model_config.decay,
        tolerance_for_calc_threshold=model_config.tolerance_for_calc_threshold,
    ).to(device)

    # 加载权重
    vqvae_path = config.training.pretrained.vqvae_path
    if os.path.exists(vqvae_path):
        checkpoint = torch.load(vqvae_path, map_location=device, weights_only=False)
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQVAE from {vqvae_path}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found: {vqvae_path}")

    vqvae_model.eval()

    return vqvae_model


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

    # 确定模型类型
    model_type = '3dcnn' if config.model.name == '3DCNN' else 'cnn'

    # 加载预训练的 VQVAE 模型
    vqvae_model = load_pretrained_vqvae(config)

    # 分割数据集
    dataset_paths = split_dataset(
        voxel_dir=ear_dir,
        hrtf_dir=hrtf_dir,
        inputform=config.dataset.input_form,
        n_folds=config.dataset.n_folds,
        val_fold=config.dataset.val_fold,
        seed=config.dataset.seed
    )

    # 创建数据集 - 使用 CNNDataSet，在内部预计算 VQVAE 特征
    train_dataset = CNNDataSet(
        hrtf_files=dataset_paths['train_hrtf_list'],
        left_voxels=dataset_paths['left_train'],
        right_voxels=dataset_paths['right_train'],
        vqvae_model=vqvae_model,
        device=device,
        status="train",
        calc_mean=False,
        use_diff=config.dataset.use_diff,
        inputform=config.dataset.input_form,
        mode=config.dataset.mode
    )

    test_dataset = CNNDataSet(
        hrtf_files=dataset_paths['test_hrtf_list'],
        left_voxels=dataset_paths['left_test'],
        right_voxels=dataset_paths['right_test'],
        vqvae_model=vqvae_model,
        device=device,
        status="test",
        calc_mean=False,
        use_diff=config.dataset.use_diff,
        inputform=config.dataset.input_form,
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

    # 创建模型
    if model_type == '3dcnn':
        model = ResNet3DClassifier(
            num_classes=config.model.num_classes,
            encoder_out_vec_num=config.model.encoder_out_vec_num
        ).to(device)
    else:
        model = ResNet2DClassifier(
            num_classes=config.model.num_classes,
            encoder_out_vec_num=config.model.encoder_out_vec_num
        ).to(device)

    print(f"Model type: {model.modelname}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
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

    # 确定使用耳朵属于左侧还是右侧
    voxel_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        train_progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            file=sys.stdout
        )

        for i, batch in enumerate(train_progress_bar):
            # 获取数据
            ear = batch[voxel_field].to(device)
            vq_indices = batch["vq_indices"].to(device)  # VQ indices as target

            optimizer.zero_grad()

            # 模型前向
            pred, logits = model(ear, device=device)

            # 损失计算 (CrossEntropy)
            loss = nn.functional.cross_entropy(logits, vq_indices)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += (pred == vq_indices).float().mean().item()

            current_lr = optimizer.param_groups[0]['lr']
            train_progress_bar.set_description(
                f"[train epoch {epoch+1}] loss: {epoch_loss/(i+1):.3f} acc: {epoch_acc/(i+1):.3f} lr: {current_lr:.2e}"
            )

        avg_loss_train = epoch_loss / len(train_loader)
        avg_acc_train = epoch_acc / len(train_loader)

        if writer is not None:
            writer.add_scalar("train_loss", avg_loss_train, epoch)
            writer.add_scalar("train_acc", avg_acc_train, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # 验证
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
            for step, batch in enumerate(val_progress_bar):
                ear_val = batch[voxel_field].to(device)
                vq_indices_val = batch["vq_indices"].to(device)

                pred_val, logits_val = model(ear_val, device=device)

                loss_val = nn.functional.cross_entropy(logits_val, vq_indices_val)
                acc_val = (pred_val == vq_indices_val).float().mean()

                val_loss += loss_val.item()
                val_acc += acc_val.item()

                val_progress_bar.set_description(
                    f"[valid epoch {epoch+1}] loss: {val_loss/(step+1):.3f} acc: {val_acc/(step+1):.3f}"
                )

        avg_loss_val = val_loss / len(test_loader)
        avg_acc_val = val_acc / len(test_loader)

        if writer is not None:
            writer.add_scalar("val_loss", avg_loss_val, epoch)
            writer.add_scalar("val_acc", avg_acc_val, epoch)

        # 更新学习率
        scheduler.step(avg_loss_train)

        print("")

        # 保存模型
        if (epoch + 1) % config.training.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None
            }
            if config.training.log:
                torch.save(checkpoint, f"{weightdir}/exp_{experiment_id:03d}_epoch_{epoch+1}.pt")
                print(f"Checkpoint saved at epoch {epoch+1}")

    if writer is not None:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
