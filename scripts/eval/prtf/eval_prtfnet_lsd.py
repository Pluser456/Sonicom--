"""
PRTFNet LSD 评估脚本 - 计算 PRTFNet 生成 HRTF 的 LSD 指标
用法:
    python scripts/eval/prtf/eval_prtfnet_lsd.py --config configs/eval/prtfnet-lsd.yaml
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config, save_config
from src.dataset.prtf import SingleSubjectDataSet
from src.utils.data import split_dataset
from src.models.prtfnet import PRTFNet


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PRTFNet LSD Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/prtfnet-lsd.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_prtfnet(config):
    """加载预训练的 PRTFNet 模型"""
    device = torch.device(config.evaluation.device)

    model = PRTFNet(
        pos_num=config.dataset.pos_num,  # 总方位数
        freq_num=config.dataset.freq_num  # 频率点数
    ).to(device)

    ckpt_path = config.pretrained.prtfnet_path
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded PRTFNet from {ckpt_path}")
    else:
        raise FileNotFoundError(f"PRTFNet checkpoint not found: {ckpt_path}")

    model.eval()
    return model


def evaluate_one_hrtf(prtfnet, test_loader, ear_field, device):
    """对单个 HRTF 样本进行推理，返回预测和真实值"""
    prtfnet.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            hrtf = batch["hrtf"].to(device)           # [batch, positions, freq_num]
            one_hot = batch["one_hot"].float().to(device)  # [batch, positions, pos_num]
            ear = batch[ear_field].to(device)         # [batch, positions, ...]

            outputs = prtfnet(ear, one_hot, device=device)
            targets = 20 * torch.log10(hrtf)
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    final_preds = torch.cat(all_preds, dim=1)  # [1, positions, freq_num]
    final_targets = torch.cat(all_targets, dim=1)

    return final_preds, final_targets


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    device = torch.device(config.evaluation.device)

    # 加载模型
    prtfnet = load_prtfnet(config)

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

    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    # ---- 逐样本 LSD 计算 ----
    res_list = []
    pred_list = []
    true_list = []

    for hrtfid in range(1, len(dataset_paths["test_hrtf_list"]) + 1):
        single_dataset = SingleSubjectDataSet(
            dataset_paths["test_hrtf_list"],
            dataset_paths["left_test"],
            dataset_paths["right_test"],
            train_log_mean_hrtf_left=None,
            train_log_mean_hrtf_right=None,
            subject_id=hrtfid,
            mode=config.dataset.mode,
            inputform=config.dataset.input_form,
            pos_num_per_batch=config.evaluation.pos_num_per_batch
        )
        single_loader = DataLoader(
            single_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=single_dataset.collate_fn
        )
        pred, true = evaluate_one_hrtf(prtfnet, single_loader, ear_field, device)
        pred = torch.abs(pred)
        true = torch.abs(true)

        pred_list.append(pred)
        true_list.append(true)

        lsd = torch.sqrt(torch.mean((pred - true) ** 2)).item()
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

    # 逐频率点的 LSD (先对受试者内的方位求均方根，再对受试者求平均)
    avg_lsd_per_freq = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((pred_tensor[:, :, freq_idx] - true_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()

    # ---- 创建结果目录 ----
    dataset_name = config.dataset.name
    run_name = f"lsd_{dataset_name}"

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

    # 保存配置
    save_config(config, result_dir / "config.yaml")

    # 保存结果
    np.savetxt(result_dir / "lsd_per_sample.txt", np.array(res_list), fmt='%.6f', header='LSD per sample (dB)')
    np.savetxt(result_dir / "lsd_per_frequency.txt", avg_lsd_per_freq, fmt='%.3f', header='LSD per frequency (dB)')
    np.savetxt(result_dir / "freq_data.txt", freq_list, fmt='%.1f', header='Frequency (Hz)')

    # 汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean LSD: {np.mean(res_list):.6f} dB\n")
        f.write(f"Number of samples: {len(res_list)}\n")
        f.write(f"Frequency bins: {len(avg_lsd_per_freq)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"\nResults saved to {result_dir}")


if __name__ == "__main__":
    main()
