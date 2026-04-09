"""
VAE-DNN-CVAE HRTF 导出脚本 - 将混合架构预测的特定方向 HRTF 导出为独立文件
用法:
    python scripts/eval/hybrid/export_hrtf.py --config configs/eval/vae-dnn-cvae-export.yaml
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.utils.config import load_config, save_config
from src.dataset.hybrid import FullPipelineDataSet
from src.utils.data import split_dataset
from src.models.hybrid.vae import VAE
from src.models.hybrid.dnn import DNN
from src.models.hybrid.cvae import CVAE


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Export VAE-DNN-CVAE HRTF for specific directions')
    parser.add_argument('--config', type=str, default='configs/eval/vae-dnn-cvae-export.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vae(config):
    """
    加载预训练的 VAE 模型（耳部图像编码器）

    Args:
        config: 配置对象，包含 pretrained.vae_path, pretrained.vae_config, evaluation.device

    Returns:
        model: 加载了权重的 VAE 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)
    vae_config = load_config(config.pretrained.vae_config)

    model = VAE(
        use_inception=vae_config.model.use_inception,
        repeat_per_block=vae_config.model.repeat_per_block,
        latent_size=vae_config.model.latent_size
    ).to(device)

    ckpt_path = config.pretrained.vae_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded VAE from {ckpt_path}")
    model.eval()
    return model


def load_pretrained_dnn(config):
    """
    加载预训练的 DNN 模型（耳部潜变量到 HRTF 潜变量的映射）

    Args:
        config: 配置对象，包含 pretrained.dnn_path, pretrained.dnn_config, evaluation.device

    Returns:
        model: 加载了权重的 DNN 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)
    dnn_config = load_config(config.pretrained.dnn_config)

    input_size = dnn_config.model.z_ears_size + 2  # z_ears + az + el
    hidden_layers = dnn_config.model.hidden_layers
    output_size = dnn_config.model.z_hrtf_size

    model = DNN(
        input_size=input_size,
        outputs_size=output_size,
        hidden_layers=hidden_layers
    ).to(device)

    ckpt_path = config.pretrained.dnn_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded DNN from {ckpt_path}")
    model.eval()
    return model


def load_pretrained_cvae(config):
    """
    加载预训练的 CVAE 模型（HRTF 解码器）

    Args:
        config: 配置对象，包含 pretrained.cvae_path, pretrained.cvae_config, evaluation.device

    Returns:
        model: 加载了权重的 CVAE 模型（评估模式）
    """
    device = torch.device(config.evaluation.device)
    cvae_config = load_config(config.pretrained.cvae_config)

    nfft = cvae_config.model.nfft
    encoder_layer_sizes = [nfft] + cvae_config.model.encoder_layer_sizes
    decoder_layer_sizes = cvae_config.model.decoder_layer_sizes + [nfft]

    model = CVAE(
        encoder_layer_sizes=encoder_layer_sizes,
        latent_size=cvae_config.model.latent_size,
        decoder_layer_sizes=decoder_layer_sizes,
        num_labels=cvae_config.model.num_labels
    ).to(device)

    ckpt_path = config.pretrained.cvae_path
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded CVAE from {ckpt_path}")
    model.eval()
    return model


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


def export_hrtf(dataloader, cvae_model, vae_model, dnn_model, config, export_indices):
    """
    推理并导出指定方位的 HRTF

    Args:
        dataloader: 测试集 DataLoader（batch_size=1）
        cvae_model: CVAE 解码器
        vae_model: VAE 耳部编码器
        dnn_model: DNN 映射网络
        config: 配置对象
        export_indices: 字典，key 为角度标识，value 为方位索引（1-based）
    """
    assert dataloader.batch_size == 1, "DataLoader 的 batch_size 必须为 1，以便逐样本处理和导出"
    device = config.evaluation.device
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

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

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if idx == config.export.hrtfid - 1:  # idx 是 0-based，hrtfid 是 1-based
                # 获取数据
                hrtf = batch["hrtf"].to(device)                    # [1, positions, nfft]
                position_deg = batch["position"].to(device)         # [1, positions, 2] (az, el) in degrees
                image = batch[ear_field].to(device)                # [1, 1, H, W]

                # 条件向量（角度）
                c = position_deg

                # VAE 编码耳部图像
                h_vae = vae_model.encoder(image)
                z_ears = vae_model.fc_mu(h_vae).unsqueeze(1).expand(-1, hrtf.size(1), -1)  # [1, positions, z_ear_size]

                # 拼接耳部潜变量和角度条件
                z_ears_c = torch.cat((z_ears, c), dim=-1)

                # DNN 预测 HRTF 潜变量
                z_hrtf = dnn_model.forward(z_ears_c)               # [1, positions, z_hrtf_size]

                # CVAE 解码得到重建的 HRTF
                hrtf_reconstructed = cvae_model.dec(z_hrtf, c)     # [1, positions, nfft]

                # 检查该 HRTF ID 是否在导出列表中
                for angle_key, idx in export_indices.items():
                    # 提取该方位的 HRTF (idx 是 1-based，转换为 0-based)
                    true_hrtf = hrtf[0, idx-1, :].cpu().numpy()
                    pred_hrtf = hrtf_reconstructed[0, idx-1, :].cpu().numpy()
                    pred_filename = output_dir / f"hrtf_pred_{angle_key}.txt"
                    np.savetxt(pred_filename, pred_hrtf, fmt='%.3f',
                            header=f'Predicted HRTF Magnitude (dB) for direction {angle_key}')
                    print(f"  已保存预测 HRTF: {pred_filename}")

                    true_filename = output_dir / f"hrtf_true_{angle_key}.txt"
                    np.savetxt(true_filename, true_hrtf, fmt='%.3f',
                            header=f'True HRTF Magnitude (dB) for direction {angle_key}')
                    print(f"  已保存真实 HRTF: {true_filename}")

    return output_dir


def main():
    args = parse_args()

    # 加载配置
    if not Path(args.config).exists():
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    # 加载模型
    vae_model = load_pretrained_vae(config)
    dnn_model = load_pretrained_dnn(config)
    cvae_model = load_pretrained_cvae(config)

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

    # 训练数据集（仅获取 mean）
    train_dataset = FullPipelineDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=config.dataset.use_diff,
        status="train",
        mode=config.dataset.mode
    )

    # 测试数据集
    test_dataset = FullPipelineDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        use_diff=config.dataset.use_diff,
        status="test",
        mode=config.dataset.mode,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    # 获取频率列表
    freq_list = get_freq_list(config.dataset.name)

    # 导出配置
    export_indices = config.export.indices
    hrtfid = config.export.hrtfid

    print(f"\n{'='*60}")
    print(f"开始导出指定方向的 HRTF...")
    print(f"数据集：{config.dataset.name}")
    print(f"模式：{config.dataset.mode}")
    print(f"HRTF 样本 ID: {hrtfid}")
    print(f"要导出的方向：{export_indices}")
    print(f"{'='*60}\n")

    # 导出 HRTF
    output_dir = export_hrtf(test_loader, cvae_model, vae_model, dnn_model, config, export_indices)

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
