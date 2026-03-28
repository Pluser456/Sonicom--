"""
DNN 训练脚本 - 使用预计算的 VAE/CVAE 潜在变量
用法:
    python train_dnn.py --config configs/vae-dnn-cvae/dnn_default.yaml
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
from src.dataset.hrtf import DNNDataSet
from src.utils.data import split_dataset
from src.models.hybrid.dnn import DNN
from src.utils.training import create_experiment
from src.models.hybrid.vae import VAE
from src.models.hybrid.cvae import CVAE


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DNN Training')
    parser.add_argument('--config', type=str, default='configs/vae-dnn-cvae/dnn_default.yaml',
                        help='Path to config file')
    parser.add_argument('--weightname', type=str, default='nopretrain',
                        help='Weight file name')
    return parser.parse_args()


def loss_function(pred, target):
    """MSE 损失函数"""
    return nn.functional.mse_loss(pred, target, reduction='mean')


def load_pretrained_models(config):
    """加载预训练的 VAE 和 CVAE 模型"""
    device = torch.device(config.training.device)

    # 加载 VAE 模型配置
    vae_config_path = config.training.pretrained.vae_config
    vae_config = load_config(vae_config_path)
    vae_model = VAE(
        use_inception=vae_config.model.use_inception,
        repeat_per_block=vae_config.model.repeat_per_block,
        latent_size=vae_config.model.latent_size
    ).to(device)
    vae_path = config.training.pretrained.vae_path
    if os.path.exists(vae_path):
        vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
        vae_model.load_state_dict(vae_checkpoint['model_state_dict'])
        print(f"Loaded VAE from {vae_path}")
    else:
        raise FileNotFoundError(f"VAE checkpoint not found: {vae_path}")
    vae_model.eval()

    # 加载 CVAE 模型配置
    cvae_config_path = config.training.pretrained.cvae_config
    cvae_config = load_config(cvae_config_path)
    nfft = cvae_config.model.nfft
    encoder_layer_sizes = [nfft] + cvae_config.model.encoder_layer_sizes
    decoder_layer_sizes = cvae_config.model.decoder_layer_sizes + [nfft]
    cvae_model = CVAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=cvae_config.model.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        num_labels=cvae_config.model.num_labels
    ).to(device)
    cvae_path = config.training.pretrained.cvae_path
    if os.path.exists(cvae_path):
        cvae_checkpoint = torch.load(cvae_path, map_location=device, weights_only=False)
        cvae_model.load_state_dict(cvae_checkpoint['model_state_dict'])
        print(f"Loaded CVAE from {cvae_path}")
    else:
        raise FileNotFoundError(f"CVAE checkpoint not found: {cvae_path}")
    cvae_model.eval()

    return vae_model, cvae_model


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

    # 加载预训练的 VAE 和 CVAE 模型
    vae_model, cvae_model = load_pretrained_models(config)

    # 加载数据集
    dataset_paths = split_dataset(ear_dir, hrtf_dir, inputform=config.dataset.input_form,
                                  n_folds=config.dataset.n_folds, val_fold=config.dataset.val_fold,
                                  seed=config.dataset.seed)

    # 创建数据集
    train_dataset = DNNDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        vae_model=vae_model,
        cvae_model=cvae_model,
        device=device,
        status="train",
        calc_mean=True,
        mode=config.dataset.mode
    )
    test_dataset = DNNDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        vae_model=vae_model,
        cvae_model=cvae_model,
        device=device,
        status="test",
        calc_mean=False,
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

    # 模型实例化
    input_size = config.model.z_ears_size + 2  # z_ears + az + el
    output_size = config.model.z_hrtf_size
    hidden_layers = config.model.hidden_layers

    model = DNN(
        input_size=input_size,
        outputs_size=output_size,
        hidden_layers=hidden_layers
    ).to(device)

    print(f"Model input: {input_size}, output: {output_size}")
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

    # 确定使用哪个字段
    z_ears_field = "z_ears_left" if config.dataset.mode == "left" else "z_ears_right"


    # 训练循环
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0

        train_progress_bar = tqdm.tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            file=sys.stdout
        )

        for i, batch in enumerate(train_progress_bar):
            # 获取数据
            z_ears = batch[z_ears_field].to(device)      # [batch, 64]
            position = batch["position"].float().to(device)  # [batch, 3]
            z_hrtf_true = batch["z_hrtf"].to(device)  # [batch, 32]

            # 拼接输入
            x = torch.cat([z_ears, position], dim=-1)

            optimizer.zero_grad()

            # 模型前向
            z_hrtf_pred = model(x)

            # 损失计算
            loss = loss_function(z_hrtf_pred, z_hrtf_true)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            current_lr = optimizer.param_groups[0]['lr']
            train_progress_bar.set_description(
                f"[train epoch {epoch+1}] loss: {epoch_loss/(i+1):.6f} lr: {current_lr:.2e}"
            )

        avg_loss_train = epoch_loss / len(train_loader)

        if writer is not None:
            writer.add_scalar("train_loss", avg_loss_train, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # 验证
        model.eval()
        val_loss = 0

        with torch.no_grad():
            val_progress_bar = tqdm.tqdm(test_loader, file=sys.stdout)
            for step, batch in enumerate(val_progress_bar):
                z_ears_val = batch[z_ears_field].to(device)
                position_val = batch["position"].float().to(device)
                z_hrtf_val = batch["z_hrtf"].to(device)

                x_val = torch.cat([z_ears_val, position_val], dim=-1)
                z_hrtf_pred_val = model(x_val)

                loss_val = loss_function(z_hrtf_pred_val, z_hrtf_val)
                val_loss += loss_val.item()

                val_progress_bar.set_description(
                    f"[valid epoch {epoch+1}] loss: {val_loss/(step+1):.6f}"
                )

        avg_loss_val = val_loss / len(test_loader)

        if writer is not None:
            writer.add_scalar("val_loss", avg_loss_val, epoch)

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
            torch.save(checkpoint, f"{weightdir}/exp_{experiment_id:03d}_epoch_{epoch+1}.pt")
            print(f"Checkpoint saved at epoch {epoch+1}")

    if writer is not None:
        writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
