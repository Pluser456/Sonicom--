"""
VQVAE 重建 LSD 评估脚本 - 仅使用 VQVAE 编码-量化-解码计算重建 LSD
用法:
    python scripts/eval/eval_reconstruct_lsd.py --config configs/eval/reconstruct-lsd-eval.yaml
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config, save_config
from src.dataset.hrtf import SonicomDataSet, SingleSubjectDataSet
from src.utils.data import split_dataset
from src.models.AE import HRTF_VQVAE

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='VQVAE Reconstruct LSD Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/reconstruct-lsd-eval.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vqvae(config):
    """加载预训练的 VQVAE 模型"""
    device = torch.device(config.evaluation.device)

    # 加载 VQVAE 配置
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


def evaluate_one_hrtf(vqvae, test_loader, usediff, device):
    """通过 VQVAE 编码-量化-解码计算重建 LSD"""
    vqvae.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            targets = batch["hrtf"]
            pos = batch["position"].to(device)

            # 转换到对数域 (dB)
            targets = targets + 1e-8
            log_target = 20 * torch.log10(targets)
            log_target_on_device = log_target.to(device)

            # VQVAE 重建 (编码-量化-解码)
            outputs, _, _ = vqvae(log_target_on_device, pos)
            outputs = outputs.squeeze(1)
            if usediff:
                meanloghrtf = batch["meanlog"].to(device)
                pred = torch.abs(outputs + meanloghrtf)
            else:
                pred = torch.abs(outputs)
            log_target = torch.abs(log_target_on_device)

            all_preds.append(pred.cpu())
            all_targets.append(log_target.cpu())

    final_preds = torch.cat(all_preds, dim=0)
    final_targets = torch.cat(all_targets, dim=0)

    return final_preds, final_targets


def main():
    args = parse_args()

    # 加载配置
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    device = torch.device(config.evaluation.device)

    # 加载模型
    vqvae = load_pretrained_vqvae(config)

    # 数据路径
    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)

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

    usediff = config.dataset.use_diff

    # 逐样本 LSD 计算
    res_list = []
    pred_list = []
    true_list = []

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
        pred_log_hrtf, true_log_hrtf = evaluate_one_hrtf(vqvae, dataloader, usediff, device)

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
        freq_list = 48000 / 240 * freq_list
    elif config.dataset.name == "sonicom":
        freq_list = np.linspace(0, 107, 108)
        freq_list = 48000 / 256 * freq_list

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
    dataset_name = config.dataset.name
    run_name = f"lsd_recon_{dataset_name}"

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
    np.savetxt(result_dir / "lsd_per_sample.txt", np.array(res_list), fmt='%.6f', header='LSD per sample (dB)')
    np.savetxt(result_dir / "lsd_per_sample_mean.txt", np.array(res_list_mean), fmt='%.6f', header='LSD of mean HRTF per sample (dB)')
    np.savetxt(result_dir / "lsd_per_frequency.txt", avg_lsd_per_freq, fmt='%.3f', header='LSD per frequency (dB)')
    np.savetxt(result_dir / "lsd_per_frequency_mean.txt", avg_lsd_per_freq_of_mean, fmt='%.3f', header='LSD per frequency of mean HRTF (dB)')
    np.savetxt(result_dir / "freq_data.txt", freq_list, fmt='%.1f', header='Frequency (Hz)')

    # 保存汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean LSD (reconstructed vs original): {np.mean(res_list):.6f} dB\n")
        f.write(f"Mean LSD (mean HRTF vs original): {np.mean(res_list_mean):.6f} dB\n")
        f.write(f"Number of samples: {len(res_list)}\n")
        f.write(f"Number of frequency bins: {len(freq_list)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
