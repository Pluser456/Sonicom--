"""
HRTF 提取脚本 - 获取指定方位的 HRTF 并保存
用法:
    python scripts/eval/hybrid/extract_hrtf.py --config configs/eval/vae-dnn-cvae-lsd.yaml
"""
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.utils.config import load_config
from src.dataset.hybrid import FullPipelineDataSet
from src.utils.data import split_dataset
from src.models.hybrid.vae import VAE
from src.models.hybrid.dnn import DNN
from src.models.hybrid.cvae import CVAE


def parse_args():
    parser = argparse.ArgumentParser(description='HRTF Extraction')
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
    ckpt = torch.load(config.pretrained.vae_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded VAE from {config.pretrained.vae_path}")
    model.eval()
    return model


def load_pretrained_dnn(config):
    device = torch.device(config.evaluation.device)
    input_size = config.model.z_ears_size + 3
    model = DNN(
        input_size=input_size,
        outputs_size=config.model.z_hrtf_size,
        hidden_layers=config.model.hidden_layers
    ).to(device)
    ckpt = torch.load(config.pretrained.dnn_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded DNN from {config.pretrained.dnn_path}")
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
    ckpt = torch.load(config.pretrained.cvae_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded CVAE from {config.pretrained.cvae_path}")
    model.eval()
    return model


def calculate_txt(dataloader, cvae_model, vae_model, dnn_model, device, save_dir):
    """提取指定方位的 HRTF 并保存为 txt 和图像"""
    idx_map = {
        "0_0": 1956,
        "0_90": 11,
        "0_80": 414,
        "90_0": 199,
        "20_54": 924
    }

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            if i in list(idx_map.values()):
                hrtf = batch["hrtf"].squeeze(0)          # [positions, nfft]
                position_deg = batch["position"].squeeze(0)  # [positions, 2]
                left_image = batch["left_voxel"].squeeze(0)   # [positions, 1, H, W]

                rad = torch.deg2rad(position_deg)
                sin_az = torch.sin(rad[:, 0])
                cos_az = torch.cos(rad[:, 0])
                sin_el = torch.sin(rad[:, 1])
                c = torch.stack([sin_az, cos_az, sin_el], dim=-1)

                h_vae = vae_model.encoder(left_image)
                z_ears, mu, logvar = vae_model._bottleneck(h_vae)
                z_ears_c = torch.cat((z_ears, c), dim=-1)
                z_hrtf = dnn_model.forward(z_ears_c)
                hrtf_reconstructed = cvae_model.dec(z_hrtf, c)

                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(abs(hrtf[0, :].cpu().numpy()), label='Original HRTF')
                plt.title('Original HRTF Curve')
                plt.legend()
                plt.subplot(1, 2, 2)
                plt.plot(abs(hrtf_reconstructed[0, :].cpu().numpy()), label='Reconstructed HRTF')
                plt.title('Reconstructed HRTF Curve')
                plt.legend()
                match_name = list(idx_map.keys())[list(idx_map.values()).index(i)]
                plt.savefig(f'{save_dir}/hrtf_comparison_{match_name}.png')
                plt.close()

                for name, idx in idx_map.items():
                    if i == idx:
                        np.savetxt(f'{save_dir}/hrtf_VAE_{name}.txt',
                                   hrtf_reconstructed[0, :].cpu().numpy(),
                                   fmt='%.1f')
                        np.savetxt(f'{save_dir}/hrtf_original_{name}.txt',
                                   hrtf[0, :].cpu().numpy(),
                                   fmt='%.1f')
                        print(f"Saved HRTF for {name}")
                        break

            if i > max(idx_map.values()):
                break


def main():
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found.")
    config = load_config(args.config)

    device = torch.device(config.evaluation.device)
    vae_model = load_pretrained_vae(config)
    dnn_model = load_pretrained_dnn(config)
    cvae_model = load_pretrained_cvae(config)

    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)
    dataset_paths = split_dataset(ear_dir, hrtf_dir, inputform=config.dataset.input_form,
                                  n_folds=config.dataset.n_folds, val_fold=config.dataset.val_fold, seed=config.dataset.seed)

    train_dataset = FullPipelineDataSet(
        dataset_paths["train_hrtf_list"], dataset_paths["left_train"], dataset_paths["right_train"],
        use_diff=config.dataset.use_diff, status="train", mode=config.dataset.mode)

    test_dataset = FullPipelineDataSet(
        dataset_paths["test_hrtf_list"], dataset_paths["left_test"], dataset_paths["right_test"],
        use_diff=config.dataset.use_diff, status="test", mode=config.dataset.mode,
        provided_mean_left=train_dataset.log_mean_hrtf_left, provided_mean_right=train_dataset.log_mean_hrtf_right)

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=test_dataset.collate_fn)

    calculate_txt(test_loader, cvae_model, vae_model, dnn_model, device, save_dir=".")


if __name__ == "__main__":
    main()