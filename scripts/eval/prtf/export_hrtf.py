"""
PRTFNet HRTF 导出脚本 - 将 PRTFNet 预测的特定方向 HRTF 导出为独立文件
用法:
    python scripts/eval/prtf/export_hrtf.py --config configs/eval/prtfnet-export-so.yaml
"""
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
    parser = argparse.ArgumentParser(description='Export PRTFNet HRTF for specific directions')
    parser.add_argument('--config', type=str, default='configs/eval/prtfnet-export-so.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_prtfnet(config):
    """
    加载预训练的 PRTFNet 模型

    Args:
        config: 配置对象，包含 pretrained.prtfnet_path, dataset.pos_num, dataset.freq_num, evaluation.device

    Returns:
        model: 加载了权重的 PRTFNet 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)

    model = PRTFNet(
        pos_num=config.dataset.pos_num,
        freq_num=config.dataset.freq_num
    ).to(device)

    ckpt_path = config.pretrained.prtfnet_path
    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded PRTFNet from {ckpt_path}")
    else:
        raise FileNotFoundError(f"PRTFNet checkpoint not found: {ckpt_path}")

    model.eval()
    return model


def infer_one_hrtf(prtfnet, test_loader, ear_field, device):
    """
    对单个 HRTF 样本进行推理，返回预测和真实值

    Args:
        prtfnet: PRTFNet 模型
        test_loader: DataLoader
        ear_field: 耳部数据字段名 ('left_voxel' or 'right_voxel')
        device: 计算设备

    Returns:
        pred_log_hrtf: 预测的 HRTF (batch, positions, freq)
        true_log_hrtf: 真实的 HRTF (batch, positions, freq)
    """
    prtfnet.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            hrtf = batch["hrtf"].to(device)
            one_hot = batch["one_hot"].float().to(device)
            ear = batch[ear_field].to(device)

            outputs = prtfnet(ear, one_hot, device=device)
            targets = 20 * torch.log10(hrtf)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    final_preds = torch.cat(all_preds, dim=1)
    final_targets = torch.cat(all_targets, dim=1)

    return final_preds, final_targets


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
        freq_list = 48000 / 240 * freq_list
    elif dataset_name == "sonicom":
        freq_list = np.linspace(0, 107, 108)
        freq_list = 48000 / 256 * freq_list
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return freq_list


def main():
    args = parse_args()

    # 加载配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    # 设置设备
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

    # 获取频率列表
    freq_list = get_freq_list(config.dataset.name)

    # 指定要导出的 HRTF ID 和角度 index 的字典
    export_indices = config.export.indices
    hrtfid = config.export.hrtfid

    print(f"\n{'='*60}")
    print(f"开始导出指定方向的 HRTF...")
    print(f"数据集：{config.dataset.name}")
    print(f"模式：{config.dataset.mode}")
    print(f"HRTF 样本 ID: {hrtfid}")
    print(f"要导出的方向：{export_indices}")
    print(f"{'='*60}\n")

    # 创建单个受试者数据集
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

    # 推理
    pred_log_hrtf, true_log_hrtf = infer_one_hrtf(prtfnet, single_loader, ear_field, device)

    # 创建输出目录
    dataset_name = config.dataset.name
    run_name = f"picked_hrtf_{dataset_name}"

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

    # 导出每个指定角度的 HRTF
    for angle_key, idx in export_indices.items():
        print(f"正在导出角度 {angle_key} (index={idx})...")

        # 获取该方向的 HRTF (索引是 1-based，数组是 0-based)
        pred_hrtf = pred_log_hrtf[0, idx-1, :].numpy()
        true_hrtf = true_log_hrtf[0, idx-1, :].numpy()

        # 保存预测的 HRTF
        pred_filename = output_dir / f"hrtf_pred_{angle_key}.txt"
        np.savetxt(pred_filename, pred_hrtf, fmt='%.3f',
                  header=f'Predicted HRTF Magnitude (dB) for direction {angle_key}')
        print(f"  已保存预测 HRTF: {pred_filename}")

        # 保存真实的 HRTF
        true_filename = output_dir / f"hrtf_true_{angle_key}.txt"
        np.savetxt(true_filename, true_hrtf, fmt='%.3f',
                  header=f'True HRTF Magnitude (dB) for direction {angle_key}')
        print(f"  已保存真实 HRTF: {true_filename}")

    # 保存配置文件副本
    save_config(config, output_dir / "config.yaml")
    print(f"\n已保存配置文件：{output_dir / 'config.yaml'}")

    # 保存频率列表
    freq_filename = output_dir / "freq_data.txt"
    np.savetxt(freq_filename, freq_list, fmt='%.1f', header='Frequency (Hz)')
    print(f"已保存频率数据：{freq_filename}")

    print(f"\n{'='*60}")
    print(f"导出完成！所有文件保存在:")
    print(f"  {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
