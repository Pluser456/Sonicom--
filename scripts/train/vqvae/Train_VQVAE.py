"""
VQVAE 训练脚本 - 支持配置文件
用法:
    python Train_VQVAE.py --config configs/default.yaml
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
from transformers import get_cosine_schedule_with_warmup

from src.utils.config import load_config
from src.dataset.vqvae import OnlyHRTFDataSet
from src.utils.data import split_dataset
from src.models.AE import HRTF_VQVAE
from src.utils.training import create_experiment

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VQVAE Training')
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

    train_dataset = OnlyHRTFDataSet(
        dataset_paths["train_hrtf_list"],
        use_diff=config.dataset.use_diff,
        calc_mean=config.dataset.use_diff,
        status="test",
        mode=config.dataset.mode
    )
    test_dataset = OnlyHRTFDataSet(
        dataset_paths["test_hrtf_list"],
        calc_mean=False,
        status="test",
        mode=config.dataset.mode,
        use_diff=config.dataset.use_diff,
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

    # 模型实例化
    model = HRTF_VQVAE(
        hrtf_row_len=config.model.hrtf_row_len,
        encoder_out_vec_num=config.model.encoder_out_vec_num,
        embed_dim=config.model.embed_dim,
        encoder_transformer_config=config.model.transformer_encoder_settings,
        decoder_transformer_config=config.model.transformer_decoder_settings,
        num_embeddings=config.model.codebook_size,
        use_VQ=config.model.use_VQ,
        input_pos_as_seq=config.model.input_pos_as_seq,
        tolerance_for_calc_threshold=config.model.tolerance_for_calc_threshold,
        decay=config.model.decay
    ).to(device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    reconstruction_loss_fn = nn.MSELoss()
    num_epochs = config.training.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.num_warmup_epochs,
        num_training_steps=num_epochs
    )

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

    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss_recon = 0
        epoch_loss_vq = 0

        train_progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            file=sys.stdout
        )
        indexes_list = []

        for i, batch in enumerate(train_progress_bar):
            hrtf = batch["hrtf"].to(device)
            pos = batch["position"].to(device)

            optimizer.zero_grad()

            reconstructed_hrtf, vq_loss, indices = model(hrtf, pos)
            indexes_list.append(indices)
            recon_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
            total_loss = recon_loss + vq_loss * config.model.commitment_cost_beta

            total_loss.backward()
            optimizer.step()

            epoch_loss_recon += recon_loss.item()
            epoch_loss_vq += vq_loss.item()

            train_progress_bar.desc = (
                f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.2f} "
                f"loss_vq: {epoch_loss_vq/(i+1):.2f} "
                f"lr: {optimizer.param_groups[0]['lr']:.2e} "
            )

        indexes = torch.cat(indexes_list, dim=0)
        activity = torch.unique(indexes).numel() / config.model.codebook_size * 100

        avg_recon_loss_train = epoch_loss_recon / len(train_loader)
        avg_vq_loss_train = epoch_loss_vq / len(train_loader)
        writer.add_scalar("train_loss_recon", avg_recon_loss_train, epoch)
        writer.add_scalar("train_loss_vq", avg_vq_loss_train, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar("activity", activity, epoch)

        # 验证
        model.eval()
        val_loss_recon = 0
        val_loss_vq = 0
        indexes_list = []

        with torch.no_grad():
            val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
            for step, batch in enumerate(val_progress_bar):
                hrtf_val = batch["hrtf"].to(device)
                pos_val = batch["position"].to(device)

                reconstructed_hrtf_val, vq_loss_val, indices = model(hrtf_val, pos_val)
                recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf_val, hrtf_val)
                indexes_list.append(indices)
                val_loss_recon += recon_loss_val.item()
                val_loss_vq += vq_loss_val.item()

                val_progress_bar.desc = (
                    f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.3f} "
                    f"loss_vq: {val_loss_vq/(step+1):.3f}"
                )

        avg_recon_loss_val = val_loss_recon / len(test_loader)
        avg_vq_loss_val = val_loss_vq / len(test_loader)
        activity_val = torch.unique(torch.cat(indexes_list, dim=0)).numel() / config.model.codebook_size * 100

        writer.add_scalar("val_loss_recon", avg_recon_loss_val, epoch)
        writer.add_scalar("val_loss_vq", avg_vq_loss_val, epoch)
        writer.add_scalar("val_activity", activity_val, epoch)
        scheduler.step()
        print("")

        # 保存模型
        if (epoch + 1) % 30 == 0:
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
