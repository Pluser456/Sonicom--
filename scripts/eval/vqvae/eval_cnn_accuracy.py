"""
CNN 分类器准确度评估脚本 - 评估 CNN 预测 VQ 索引的准确率
用法:
    python scripts/eval/eval_cnn_accuracy.py --config configs/eval/cnn-accuracy-eval.yaml
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.config import load_config, save_config
from src.models.TestNet import ResNet3DClassifier, ResNet2DClassifier
from src.models.AE import HRTF_VQVAE
from src.dataset.vqvae import CNNDataSet
from src.dataset.hrtf import SonicomDataSet
from src.utils.data import split_dataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CNN Accuracy Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/cnn-accuracy-eval.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vqvae(config):
    """加载预训练的 VQVAE 模型（用于获取真实 VQ 索引）"""
    device = torch.device(config.evaluation.device)

    vqvae_config = load_config(config.pretrained.vqvae_config)
    model_config = vqvae_config.model

    vqvae = HRTF_VQVAE(
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

    vqvae_ckpt = config.pretrained.vqvae_path
    if os.path.exists(vqvae_ckpt):
        checkpoint = torch.load(vqvae_ckpt, map_location=device, weights_only=False)
        vqvae.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQVAE from {vqvae_ckpt}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found: {vqvae_ckpt}")

    vqvae.eval()
    return vqvae


def load_pretrained_cnn(config):
    """加载预训练的 CNN 模型"""
    device = torch.device(config.evaluation.device)

    cnn_config = load_config(config.pretrained.cnn_config)
    cnn_model_config = cnn_config.model

    model_type = config.cnn.model_type
    if model_type == "3DResNet":
        model = ResNet3DClassifier(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    elif model_type == "2DResNet":
        model = ResNet2DClassifier(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    cnn_ckpt = config.pretrained.cnn_path
    if os.path.exists(cnn_ckpt):
        checkpoint = torch.load(cnn_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CNN from {cnn_ckpt}")
    else:
        raise FileNotFoundError(f"CNN checkpoint not found: {cnn_ckpt}")

    model.eval()
    return model


def main():
    args = parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    device = torch.device(config.evaluation.device)

    # 加载模型
    vqvae = load_pretrained_vqvae(config)
    model = load_pretrained_cnn(config)

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

    # 使用 SonicomDataSet 获取训练集 mean（不涉及 VQVAE 预计算，速度快）
    train_dataset = SonicomDataSet(
        hrtf_files=dataset_paths["train_hrtf_list"],
        left_voxels=dataset_paths["left_train"],
        right_voxels=dataset_paths["right_train"],
        use_diff=config.dataset.use_diff,
        calc_mean=True,
        status="test",
        inputform=config.dataset.input_form,
        mode=config.dataset.mode
    )

    # 加载测试集
    test_dataset = CNNDataSet(
        hrtf_files=dataset_paths["test_hrtf_list"],
        left_voxels=dataset_paths["left_test"],
        right_voxels=dataset_paths["right_test"],
        vqvae_model=vqvae,
        device=device,
        status="test",
        calc_mean=False,
        use_diff=config.dataset.use_diff,
        inputform=config.dataset.input_form,
        mode=config.dataset.mode,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # 确定使用耳朵属于左侧还是右侧
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    # 评估
    model.eval()
    with torch.no_grad():
        total_acc_sum = 0
        total_samples = 0
        acc_list = []

        for batch in tqdm(test_loader, desc="Evaluating"):
            ear = batch[ear_field].to(device)
            true_indices = batch["vq_indices"].to(device)  # [batch_size, encoder_out_vec_num]

            pred, _ = model(ear, device=device)

            acc = (pred == true_indices).float().mean().item()
            total_acc_sum += acc * ear.shape[0]
            total_samples += ear.shape[0]
            acc_list.append(acc)

    mean_acc = total_acc_sum / total_samples
    print(f"\nMean Accuracy: {mean_acc:.4f}")

    # ---- 创建结果目录 ----
    dataset_name = config.dataset.name
    run_name = f"cnn_accuracy_{dataset_name}"

    result_base = Path(config.paths.result_dir) / run_name
    result_base.mkdir(parents=True, exist_ok=True)

    existing_dirs = [d for d in result_base.iterdir() if d.is_dir() and d.name.startswith('res_')]
    res_numbers = []
    for d in existing_dirs:
        try:
            res_numbers.append(int(d.name.split('_')[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(res_numbers) + 1 if res_numbers else 0
    result_dir = result_base / f"res_{next_num:03d}"
    result_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置文件副本
    save_config(config, result_dir / "config.yaml")

    # 保存结果
    np.savetxt(result_dir / "accuracy_per_batch.txt", np.array(acc_list), fmt='%.6f', header='Accuracy per batch')

    # 保存汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean Accuracy: {mean_acc:.6f}\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Number of batches: {len(acc_list)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()