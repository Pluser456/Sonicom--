"""
CNN-VQVAE 评估通用工具函数
提供模型加载、推理等共享功能
"""
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config
from src.models.TestNet import ResNet3DClassifier as threeDResnet
from src.models.TestNet import ResNet2DClassifier as twoDResnet
from src.dataset.hrtf import SonicomDataSet, SingleSubjectDataSet
from src.utils.data import split_dataset
from src.models.AE import HRTF_VQVAE


def load_pretrained_vqvae(config):
    """
    加载预训练的 VQVAE 模型

    Args:
        config: 配置对象，包含 pretrained.vqvae_path, pretrained.vqvae_config, evaluation.device

    Returns:
        hrtf_encoder: 加载了权重的 VQVAE 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)

    # 加载 VQVAE 模型配置
    vqvae_config = load_config(config.pretrained.vqvae_config)
    model_config = vqvae_config.model

    # 创建模型
    hrtf_encoder = HRTF_VQVAE(
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
    vqvae_ckpt = config.pretrained.vqvae_path
    if os.path.exists(vqvae_ckpt):
        checkpoint = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
        hrtf_encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQVAE from {vqvae_ckpt}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found: {vqvae_ckpt}")

    hrtf_encoder.eval()
    return hrtf_encoder


def load_pretrained_cnn(config):
    """
    加载预训练的 CNN 分类器模型

    Args:
        config: 配置对象，包含 pretrained.cnn_path, pretrained.cnn_config, cnn.model_type, evaluation.device

    Returns:
        model: 加载了权重的 CNN 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)

    cnn_config = load_config(config.pretrained.cnn_config)
    cnn_model_config = cnn_config.model

    model_type = config.cnn.model_type
    if model_type == "3DResNet":
        model = threeDResnet(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    elif model_type == "2DResNet":
        model = twoDResnet(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cnn_ckpt = config.pretrained.cnn_path
    if os.path.exists(cnn_ckpt):
        checkpoint = torch.load(cnn_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CNN from {cnn_ckpt}")
    else:
        raise FileNotFoundError(f"CNN checkpoint not found: {cnn_ckpt}")

    model.eval()
    return model


def prepare_vqvae_dataset(config):
    """
    准备 VQVAE 评估数据集

    Args:
        config: 配置对象，包含 dataset, paths 等配置

    Returns:
        dataset_paths: 分割后的数据集路径字典
        train_dataset: 训练数据集（用于获取 mean HRTF）
        log_mean_hrtf_left: 左耳平均 HRTF
        log_mean_hrtf_right: 右耳平均 HRTF
    """
    # 数据路径
    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)

    # 分割数据集
    dataset_paths = split_dataset(
        ear_dir, hrtf_dir,
        inputform=config.dataset.input_form,
        n_folds=config.dataset.n_folds,
        val_fold=config.dataset.val_fold,
        seed=config.dataset.seed
    )

    # 训练数据集（用于计算 mean）
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=config.dataset.use_diff,
        calc_mean=True,
        status="test",
        inputform=config.dataset.input_form,
        mode=config.dataset.mode
    )
    log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
    log_mean_hrtf_right = train_dataset.log_mean_hrtf_right

    return dataset_paths, train_dataset, log_mean_hrtf_left, log_mean_hrtf_right


def create_single_subject_dataloader(dataset_paths, hrtfid, config,
                                    log_mean_hrtf_left=None, log_mean_hrtf_right=None):
    """
    创建单个受试者的 DataLoader

    Args:
        dataset_paths: 数据集路径字典
        hrtfid: 受试者 ID（1-based）
        config: 配置对象
        log_mean_hrtf_left/right: 平均 HRTF（如果为 None，则从 config 重新计算）

    Returns:
        dataloader: DataLoader 对象
    """
    # 如果未提供 mean HRTF，则需要准备数据集获取
    if log_mean_hrtf_left is None or log_mean_hrtf_right is None:
        _, _, log_mean_hrtf_left, log_mean_hrtf_right = prepare_vqvae_dataset(config)

    val_dataset = SingleSubjectDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        mode=config.dataset.mode,
        train_log_mean_hrtf_left=log_mean_hrtf_left,
        train_log_mean_hrtf_right=log_mean_hrtf_right,
        subject_id=hrtfid,
        inputform=config.dataset.input_form
    )

    dataloader = DataLoader(
        val_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )
    return dataloader


def infer_one_hrtf(cnnmodel, vqvae, test_loader, usediff, ear_field, device):
    """
    对单个 HRTF 样本进行推理

    Args:
        cnnmodel: CNN 分类器模型
        vqvae: VQVAE 模型
        test_loader: DataLoader
        usediff: 是否使用差分数据
        ear_field: 耳部数据字段名 ('left_voxel' or 'right_voxel')
        device: 计算设备

    Returns:
        pred_log_hrtf: 预测的 HRTF (batch, rows, freq)
        true_log_hrtf: 真实的 HRTF (batch, rows, freq)
    """
    cnnmodel.eval()
    vqvae.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            targets = batch["hrtf"]
            meanloghrtf = batch["meanlog"].to(device)
            pos = batch["position"].to(device)
            ear = batch[ear_field].to(device)

            # CNN 预测 VQ 索引
            pred, _ = cnnmodel(ear, device=device)

            # VQVAE 解码
            zq_list = []
            for i in range(vqvae.encoder_out_vec_num):
                zq_i = vqvae.vq_layer[i].get_output_from_indices(pred[:, i])
                zq_list.append(zq_i)
            zq = torch.stack(zq_list, dim=1)
            outputs = vqvae.decoder(zq, pos)

            # 处理目标 HRTF
            targets = targets + 1e-8
            log_target = 20 * torch.log10(targets)

            # 处理预测 HRTF
            if usediff:
                pred = outputs + meanloghrtf
            else:
                pred = outputs

            all_preds.append(pred)
            all_targets.append(log_target)

    final_preds = torch.cat(all_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    return final_preds.cpu(), final_targets.cpu()


def get_freq_list(dataset_name):
    """
    根据数据集名称获取频率列表

    Args:
        dataset_name: 数据集名称 ("widespread" or "sonicom")

    Returns:
        freq_list: numpy array, 频率值列表
    """
    if dataset_name == "widespread":
        freq_list = np.linspace(0, 89, 90)
        freq_list = 48000 / 240 * freq_list  # 转换为实际频率值
    elif dataset_name == "sonicom":
        freq_list = np.linspace(0, 107, 108)
        freq_list = 48000 / 256 * freq_list  # 计算频率值
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return freq_list