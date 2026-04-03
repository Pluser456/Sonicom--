"""
训练和评估工具函数
"""
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import sys
from tqdm import tqdm
import torch
import torch.nn as nn
from omegaconf import OmegaConf


def create_experiment(log_dir: str, config, start_epoch: int) -> tuple:
    """
    创建或继续实验文件夹并保存配置

    Args:
        log_dir: 日志根目录
        config: OmegaConf 配置对象
        start_epoch: 起始 epoch，用于断点续训时设置 tensorboard 的 purge_step

    Returns:
        (exp_folder, writer): 实验文件夹路径、TensorBoard writer
    """
    log_path = Path(log_dir)
    purge_step = start_epoch

    continue_exp = config.training.continue_exp if hasattr(config.training, 'continue_exp') else None
    if continue_exp is not None:
        exp_folder = log_path / continue_exp
        if not exp_folder.exists():
            raise ValueError(f"Experiment folder {exp_folder} does not exist.")
        print(f"Continuing experiment from {exp_folder}")
        next_num = int(continue_exp.split('_')[1])

        print(f"Will purge steps >= {purge_step}")
        # 创建 SummaryWriter，传入 purge_step
        writer = SummaryWriter(log_dir=str(exp_folder), purge_step=purge_step)
    else:
        log_path.mkdir(parents=True, exist_ok=True)

        # 找到下一个可用的 exp_xxx 文件夹
        existing_dirs = [d for d in log_path.iterdir() if d.is_dir() and d.name.startswith('exp_')]
        exp_numbers = []
        for d in existing_dirs:
            try:
                exp_numbers.append(int(d.name.split('_')[1]))
            except (IndexError, ValueError):
                pass
        next_num = max(exp_numbers) + 1 if exp_numbers else 0
        exp_folder = log_path / f"exp_{next_num:03d}"

    writer = SummaryWriter(log_dir=str(exp_folder))
    # 保存当前配置到实验文件夹
    config_path = exp_folder / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        OmegaConf.save(config, f)

    return next_num, writer


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """训练一个epoch"""
    model.train()
    loss_function = nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)
    accuracy = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)  # 显示进度条

    for step, sample_batch in enumerate(data_loader):
        if model.modelname == "3DResNetANP":
            # Extract data (ensure keys match your DataLoader output)
            left_voxel = sample_batch["left_voxel"]
            right_voxel = sample_batch["right_voxel"]
            # pos = sample_batch["position"]
            # hrtf = sample_batch["hrtf"]
            feature = sample_batch["feature"]

            # Model now returns prediction and target
            mu, target_y_sel = model(left_voxel, right_voxel, pos, hrtf, device=device, is_training=True, auxiliary_data=None)

            # Ensure target is on the correct device
            target_y_sel = target_y_sel.to(device)

            loss = loss_function(mu, target_y_sel)
        elif model.modelname == "3DResNetClassifier":
            loss_function = nn.CrossEntropyLoss()
            right_voxel = sample_batch["right_voxel"]
            feature = sample_batch["feature"] # (batch_size, encoder_out_vec_num)
            pred, logits = model(right_voxel, device=device)
            loss = loss_function(logits, feature)
            accuracy += (pred == feature).float().mean()
        elif model.modelname == "3DResNet":
            left_voxel = sample_batch["left_voxel"]
            right_voxel = sample_batch["right_voxel"]
            pos = sample_batch["position"]
            hrtf = sample_batch["hrtf"]

            mu, target_y_sel = model(left_voxel, right_voxel, pos, hrtf, device=device)
            loss = loss_function(mu, target_y_sel)
        elif model.modelname == "2DResNetClassifier":
            loss_function = nn.CrossEntropyLoss()
            # left_voxel = sample_batch["left_voxel"]
            right_voxel = sample_batch["right_voxel"]
            feature = sample_batch["feature"]
            # feature = feature.reshape(feature.shape[0], -1)[:, 0]

            pred, logits = model(right_voxel, device=device)
            loss = loss_function(logits, feature)
            accuracy += (pred == feature).float().mean()

        accu_loss += loss.detach()

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        data_loader.desc = "[train epoch {}] loss: {:.3f} acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1), accuracy.item() / (step + 1))

        optimizer.step()
        optimizer.zero_grad()

    final_loss = accu_loss.item() / (step + 1)

    return final_loss, accuracy.item() / (step + 1)


def evaluate(model, data_loader, device, epoch, auxiliary_loader=None):
    """评估模型"""
    model.eval()
    loss_function = nn.MSELoss()
    accu_loss = torch.zeros(1).to(device)
    tot_accuracy = torch.zeros(1).to(device)
    data_loader = tqdm(data_loader, file=sys.stdout)

    with torch.no_grad():
        for step, sample_batch in enumerate(data_loader):
            if model.modelname == "3DResNetANP":
                left_voxel = sample_batch["left_voxel"]
                right_voxel = sample_batch["right_voxel"]
                pos = sample_batch["position"]
                hrtf = sample_batch["hrtf"]
                auxiliary_batch = next(iter(auxiliary_loader))

                mu, _ = model(left_voxel, right_voxel, pos, hrtf, device=device, is_training=False, auxiliary_data=auxiliary_batch)

                target = hrtf.to(device).squeeze(0)

                loss = loss_function(mu, target)
            elif model.modelname == "2DResNetClassifier":
                loss_function = nn.CrossEntropyLoss()
                # left_voxel = sample_batch["left_voxel"]
                right_voxel = sample_batch["right_voxel"]
                # pos = sample_batch["position"]
                # hrtf = sample_batch["hrtf"]
                feature = sample_batch["feature"]
                # feature = feature.reshape(feature.shape[0], -1)[:, 0]

                preds, logits = model(right_voxel, device=device)

                accuracy = (preds == feature).float().mean()
                loss = loss_function(logits, feature)

                tot_accuracy += accuracy.detach()
            elif model.modelname == "3DResNetClassifier":
                loss_function = nn.CrossEntropyLoss()
                right_voxel = sample_batch["right_voxel"]
                feature = sample_batch["feature"]
                # feature = feature.reshape(feature.shape[0], -1)[:, 0]
                preds, logits = model(right_voxel, device=device)
                accuracy = (preds == feature).float().mean()
                loss = loss_function(logits, feature)
                tot_accuracy += accuracy.detach()
            elif model.modelname == "3DResNet":
                left_voxel = sample_batch["left_voxel"]
                right_voxel = sample_batch["right_voxel"]
                pos = sample_batch["position"]
                hrtf = sample_batch["hrtf"]

                mu, target_y_sel = model(left_voxel, right_voxel, pos, hrtf, device=device)
                loss = loss_function(mu, target_y_sel)

            elif model.modelname == "2DResNet":
                left_voxel = sample_batch["left_voxel"]
                right_voxel = sample_batch["right_voxel"]
                pos = sample_batch["position"]
                # hrtf = sample_batch["hrtf"]
                feature = sample_batch["feature"]

                mu, target_y_sel = model(left_voxel, right_voxel, feature, device=device)
                loss = loss_function(mu, target_y_sel)
            accu_loss += loss.detach()
            data_loader.desc = "[valid epoch {}] loss: {:.3f} acc: {:.3f}".format(epoch, accu_loss.item() / (step + 1), tot_accuracy.item() / (step + 1))

    final_loss = accu_loss.item() / (step + 1)
    return final_loss, tot_accuracy.item() / (step + 1)
