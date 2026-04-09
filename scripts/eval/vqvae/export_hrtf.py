"""
HRTF 导出脚本 - 将 CNN-VQVAE 预测的特定方向 HRTF 导出为独立文件
用法:
    python scripts/eval/vqvae/export_hrtf.py --config configs/eval/vqvae-export.yaml
"""
import argparse
from pathlib import Path
import torch
import numpy as np

from src.utils.config import load_config, save_config
from src.models.utils import (
    load_pretrained_vqvae,
    load_pretrained_cnn,
    prepare_vqvae_dataset,
    create_single_subject_dataloader,
    infer_one_hrtf,
    get_freq_list
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Export HRTF for specific directions')
    parser.add_argument('--config', type=str, default='configs/eval/vqvae-export.yaml',
                        help='Path to config file')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    # 设置设备
    device = torch.device(config.evaluation.device)

    # 加载模型
    cnnmodel = load_pretrained_cnn(config)
    vqvae = load_pretrained_vqvae(config)

    # 准备数据集
    dataset_paths, train_dataset, log_mean_hrtf_left, log_mean_hrtf_right = prepare_vqvae_dataset(config)

    # 推理参数
    usediff = config.dataset.use_diff
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    # 获取频率列表
    freq_list = get_freq_list(config.dataset.name)

    # 指定要导出的 HRTF ID 和角度 index 的字典
    export_indices = config.export.indices

    print(f"\n{'='*60}")
    print(f"开始导出指定方向的 HRTF...")
    print(f"数据集：{config.dataset.name}")
    print(f"模式：{config.dataset.mode}")
    print(f"要导出的方向：{export_indices}")
    print(f"{'='*60}\n")

    hrtfid = config.export.hrtfid
    print(f"\n正在处理 HRTF 样本 {hrtfid}...")

    # 创建 DataLoader
    dataloader = create_single_subject_dataloader(
        dataset_paths, hrtfid, config,
        log_mean_hrtf_left, log_mean_hrtf_right
    )

    # 推理
    pred_log_hrtf, true_log_hrtf = infer_one_hrtf(
        cnnmodel, vqvae, dataloader, usediff, ear_field, device
    )

    # 创建输出目录
    dataset_name = config.dataset.name
    run_name = "picked_hrtf"
    run_name += "_2D" if config.cnn.model_type == "2DResNet" else "_3D"
    run_name += f"_{dataset_name}"

    result_base = Path(config.paths.output_dir) / run_name
    result_base.mkdir(parents=True, exist_ok=True)

    existing_dirs = [d for d in result_base.iterdir() if d.is_dir() and d.name.startswith('res_')]
    res_numbers = []
    for d in existing_dirs:
        try:
            res_numbers.append(int(d.name.split('_')[1]))
        except (IndexError, ValueError):
            pass
    next_num = max(res_numbers) + 1 if res_numbers else 0
    output_dir = result_base / f"res_{next_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 检查该 HRTF ID 是否在导出列表中
    for angle_key, idx in export_indices.items():

        # 获取该方向的 HRTF
        pred_hrtf = pred_log_hrtf[0].numpy()
        true_hrtf = true_log_hrtf[0].numpy()

        # 保存预测和真实的 HRTF
        pred_filename = output_dir / f"hrtf_pred_{angle_key}.txt"
        np.savetxt(pred_filename, pred_hrtf[idx-1,:], fmt='%.3f',
                    header=f'Predicted HRTF Magnitude (dB) for direction {angle_key}')
        print(f"    已保存预测 HRTF: {pred_filename}")

        true_filename = output_dir / f"hrtf_true_{angle_key}.txt"
        np.savetxt(true_filename, true_hrtf[idx-1,:], fmt='%.3f',
                    header=f'True HRTF Magnitude (dB) for direction {angle_key}')
        print(f"    已保存真实 HRTF: {true_filename}")

    # 保存配置文件副本和频率列表
    save_config(config, output_dir / "config.yaml")

    freq_filename = output_dir / "freq_data.txt"
    np.savetxt(freq_filename, freq_list, fmt='%.1f', header='Frequency (Hz)')

    print(f"\n{'='*60}")
    print(f"导出完成！所有文件保存在:")
    print(f"  {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
