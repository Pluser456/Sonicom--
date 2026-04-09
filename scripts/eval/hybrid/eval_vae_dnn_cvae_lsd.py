"""
VAE-DNN-CVAE LSD 评估脚本 - 计算混合架构生成 HRTF 的 LSD 指标
用法:
    python scripts/eval/hybrid/eval_vae_dnn_cvae_lsd.py --config configs/eval/vae-dnn-cvae-lsd.yaml
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from src.utils.config import load_config, save_config
from src.dataset.hybrid import FullPipelineDataSet
from src.utils.data import split_dataset
from src.models.hybrid.vae import VAE
from src.models.hybrid.dnn import DNN
from src.models.hybrid.cvae import CVAE


def parse_args():
    parser = argparse.ArgumentParser(description='VAE-DNN-CVAE LSD Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/vae-dnn-cvae-lsd.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vae(config):
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


def evaluate_lsd(dataloader, cvae_model, vae_model, dnn_model, config):
    """逐样本评估 LSD"""
    lsd_list = []
    hrtf_tensor_list = []
    pred_hrtf_tensor_list = []
    ear_field = "right_voxel" if config.dataset.mode == "right" else "left_voxel"

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating LSD"):
            hrtf = batch["hrtf"].to(config.evaluation.device)                    # [1, positions, nfft]
            position_deg = batch["position"].to(config.evaluation.device)         # [1, positions, 2] (az, el) in degrees
            image = batch[ear_field].to(config.evaluation.device)         # [1, 1, H, W]

            c = position_deg

            # VAE 编码
            h_vae = vae_model.encoder(image)
            z_ears = vae_model.fc_mu(h_vae).unsqueeze(1).expand(-1, hrtf.size(1), -1)  # [1, positions, z_ear_size]
            z_ears_c = torch.cat((z_ears, c), dim=-1)

            # DNN 预测
            z_hrtf = dnn_model.forward(z_ears_c)

            # CVAE 解码
            hrtf_reconstructed = cvae_model.dec(z_hrtf, c)

            hrtf_tensor_list.append(hrtf.cpu())
            pred_hrtf_tensor_list.append(hrtf_reconstructed.cpu())

            lsd = torch.sqrt(torch.mean((torch.abs(hrtf) - torch.abs(hrtf_reconstructed)) ** 2)).item()
            lsd_list.append(lsd)

    average_lsd = sum(lsd_list) / len(lsd_list)

    # 逐频率点 LSD (参考 eval_lsd.py: 先对每个受试者内的方位求均方根，再对受试者求平均)
    hrtf_tensor = torch.cat(hrtf_tensor_list, dim=0)       # [num_subjects, positions, nfft]
    pred_hrtf_tensor = torch.cat(pred_hrtf_tensor_list, dim=0)

    if config.dataset.name == "widespread":
        freq_list = np.linspace(0, 89, 90)
    elif config.dataset.name == "sonicom":
        freq_list = np.linspace(0, 107, 108)

    avg_lsd_per_freq = np.zeros(len(freq_list))
    for freq_idx in range(len(freq_list)):
        LSDvec = torch.sqrt(torch.mean((pred_hrtf_tensor[:, :, freq_idx] - hrtf_tensor[:, :, freq_idx]) ** 2, dim=1))
        avg_lsd_per_freq[freq_idx] = torch.mean(LSDvec).item()

    return average_lsd, avg_lsd_per_freq, lsd_list


def main():
    args = parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    device = torch.device(config.evaluation.device)

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

    # LSD 评估
    mean_lsd, avg_lsd_per_freq, lsd_per_sample = evaluate_lsd(
        test_loader, cvae_model, vae_model,
        dnn_model, config
    )
    print(f"Mean LSD: {mean_lsd:.6f}")

    # 创建结果目录
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
    np.savetxt(result_dir / "lsd_per_sample.txt", np.array(lsd_per_sample), fmt='%.6f', header='LSD per sample (dB)')
    np.savetxt(result_dir / "lsd_per_frequency.txt", np.array(avg_lsd_per_freq), fmt='%.3f', header='LSD per frequency (dB)')

    # 保存频率值
    # 频率计算
    if config.dataset.name == "widespread":
        freq_list = np.linspace(0, 89, 90)
        freq_list = 48000 / 240 * freq_list  # 转换为实际频率值
    elif config.dataset.name == "sonicom":
        freq_list = np.linspace(0, 107, 108)
        freq_list = 48000 / 256 * freq_list  # 计算频率值
    np.savetxt(result_dir / "freq_data.txt", freq_list, fmt='%.1f', header='Frequency (Hz)')

    # 汇总统计
    with open(result_dir / "summary.txt", 'w', encoding='utf-8') as f:
        f.write(f"Mean LSD: {mean_lsd:.6f} dB\n")
        f.write(f"Number of samples: {len(lsd_per_sample)}\n")
        f.write(f"Frequency bins: {len(avg_lsd_per_freq)}\n")
        f.write(f"Config used: {args.config}\n")

    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main()
