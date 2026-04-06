"""
LSD 评估脚本 - 计算 CNN-VQVAE 生成 HRTF 的 LSD 指标
用法:
    python scripts/eval/eval_lsd.py --config configs/eval/lsd-eval.yaml
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config, save_config
from src.models.TestNet import ResNet3DClassifier as threeDResnet
from src.models.TestNet import ResNet2DClassifier as twoDResnet
from src.dataset.hrtf import SonicomDataSet, SingleSubjectDataSet
from src.utils.data import split_dataset
from src.models.AE import HRTF_VQVAE

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LSD Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/lsd-eval.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vqvae(config):
    """加载预训练的 VQVAE 模型"""
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
    """加载预训练的 CNN 模型"""
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


def evaluate_one_hrtf(cnnmodel, vqvae, test_loader, usediff, ear_field, device):
    """对单个 HRTF 样本进行推理，返回预测和真实值"""
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
            pred, _ = cnnmodel(ear, device=device)

            zq_list = []
            for i in range(vqvae.encoder_out_vec_num):
                zq_i = vqvae.vq_layer[i].get_output_from_indices(pred[:, i])
                zq_list.append(zq_i)
            zq = torch.stack(zq_list, dim=1)
            outputs = vqvae.decoder(zq, pos)

            targets = targets + 1e-8
            log_target = 20 * torch.log10(targets)
            if usediff:
                pred = outputs + meanloghrtf
            else:
                pred = outputs

            all_preds.append(pred)
            all_targets.append(log_target)

    final_preds = torch.cat(all_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    return final_preds.cpu(), final_targets.cpu()


def main():
    args = parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    # 设置设备
    device = torch.device(config.evaluation.device)

    # 加载模型
    cnnmodel = load_pretrained_cnn(config)
    vqvae = load_pretrained_vqvae(config)

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

    # 逐样本 LSD 计算
    res_list = []
    pred_list = []
    true_list = []

    usediff = config.dataset.use_diff
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    for hrtfid in range(1, len(dataset_paths["test_hrtf_list"]) + 1):
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
        pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(cnnmodel, vqvae, dataloader,usediff, ear_field, device)
        pred_log_hrtf, true_log_hrtf = torch.abs(pred_log_hrtf), torch.abs(true_log_hrtf)

        pred_list.append(pred_log_hrtf)
        true_list.append(true_log_hrtf)

        lsd = torch.sqrt(torch.mean((pred_log_hrtf - true_log_hrtf) ** 2)).item()
        res_list.append(lsd)
        print(f"LSD of HRTF {hrtfid}: {lsd}")

    print(f"Mean LSD: {np.mean(res_list)}")

    pred_tensor = torch.cat(pred_list, dim=0)
    true_tensor = torch.cat(true_list, dim=0)

    # 频率计算
    if config.dataset.name == "widespread":
        freq_list = np.linspace(0, 89, 90)
        freq_list = 48000 / 240 * freq_list  # 转换为实际频率值
    elif config.dataset.name == "sonicom":
        freq_list = np.linspace(0, 107, 108)
        freq_list = 48000 / 256 * freq_list  # 计算频率值

    # 逐频率点的 LSD
    avg_lsd_per_freq = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()

    # ---- 与 mean HRTF 的对比 ----
    print("\n-----------------contrast with mean HRTF-----------------\n")
    res_list_mean = []
    log_mean_hrtf_right = torch.tensor(np.abs(log_mean_hrtf_right), dtype=torch.float32).cpu()
    log_mean_hrtf_right = log_mean_hrtf_right.unsqueeze(0)

    for hrtfid in range(1, len(dataset_paths["test_hrtf_list"]) + 1):
        lsd_of_mean = torch.sqrt(torch.mean((log_mean_hrtf_right - true_tensor[hrtfid - 1, :, :]) ** 2)).item()
        res_list_mean.append(lsd_of_mean)
        print(f"LSD between mean HRTF and HRTF {hrtfid}: {lsd_of_mean}")

    print(f"Mean LSD of mean HRTF: {np.mean(res_list_mean)}")

    avg_lsd_per_freq_of_mean = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((log_mean_hrtf_right[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq_of_mean[freq_idx] = torch.mean(LSDvec).item()

    # ---- 创建结果目录 ----
    input_type = '2D' if config.cnn.model_type in ['2DResNet', '2DResNetANP'] else '3D'
    dataset_name = config.dataset.name
    run_name = f"lsd_{input_type}_{dataset_name}"

    result_base = Path(config.paths.result_dir) / run_name
    result_base.mkdir(parents=True, exist_ok=True)

    # 像 create_experiment 一样创建 res_xxx 文件夹
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
    np.savetxt(result_dir / "lsd_per_sample.txt", np.array(res_list), fmt='%.6f', header='LSD per sample (dB)')
    np.savetxt(result_dir / "lsd_per_sample_mean.txt", np.array(res_list_mean), fmt='%.6f', header='LSD of mean HRTF per sample (dB)')
    np.savetxt(result_dir / "lsd_per_frequency.txt", avg_lsd_per_freq, fmt='%.3f', header='LSD per frequency (dB)')
    np.savetxt(result_dir / "lsd_per_frequency_mean.txt", avg_lsd_per_freq_of_mean, fmt='%.3f', header='LSD per frequency of mean HRTF (dB)')
    np.savetxt(result_dir / "freq_data.txt", freq_list, fmt='%.1f', header='Frequency (Hz)')

    # 保存汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean LSD (predicted vs true): {np.mean(res_list):.6f} dB\n")
        f.write(f"Mean LSD (mean HRTF vs true): {np.mean(res_list_mean):.6f} dB\n")
        f.write(f"Number of samples: {len(res_list)}\n")
        f.write(f"Number of frequency bins: {len(freq_list)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
