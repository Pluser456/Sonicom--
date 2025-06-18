import os
import math
import json
import torch
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from PIL import Image
from scipy.fft import rfftfreq
from torchvision.transforms import Compose, ToTensor, Grayscale, Resize
from vae_incept_cfg import InceptionVAECfg as VAECfg
from dnn_cfg import DNNCfg
from cvae_dense_cfg import CVAECfg
from new_dataset import SonicomDataSetLSD
from utils import split_dataset, train_one_epoch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def calculate_lsd(dataloader, cvae_model, vae_model, dnn_model, device):
    total_loss = 0.0
    mse_loss = nn.MSELoss()
    with torch.no_grad():  # 不需要计算梯度
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            hrtf = batch["hrtf"].to(device)
            sin_azimuth = batch["sin(azimuth)"].to(device)
            cos_azimuth = batch["cos(azimuth)"].to(device)
            sin_elevation = batch["sin(elevation)"].to(device)
            left_image = batch["left_image"].to(device)
            c = torch.stack([sin_azimuth, cos_azimuth, sin_elevation], dim=-1).float()

            h_vae=vae_model.vae.encoder(left_image)
            z_ears, mu, logvar = vae_model.vae._bottleneck(h_vae)
            z_ears_c = torch.cat((z_ears, c), dim=-1)
            z_hrtf = dnn_model.forward(z_ears_c)
            hrtf_reconstructed = cvae_model.cvae.dec(z_hrtf,c)

            loss = mse_loss(hrtf, hrtf_reconstructed)
            total_loss += loss.item()  # 累加损失
            print(f"Batch {i + 1}: Loss = {loss.item():.4f}, Total Loss = {total_loss / (i + 1):.4f}")

            # 打印原始hrtf曲线
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(hrtf.cpu().numpy(), label='Original HRTF')
            plt.title('Original HRTF Curve')
            plt.legend()

            # 打印重构的hrtf曲线
            plt.subplot(1, 2, 2)
            plt.plot(hrtf_reconstructed.cpu().numpy(), label='Reconstructed HRTF')
            plt.title('Reconstructed HRTF Curve')
            plt.legend()

            plt.show()
            break  # 处理完第一个batch后退出循环


    # 计算平均损失
    average_loss = total_loss / len(dataloader)
    print(f"Average MSE Loss: {average_loss}")
    return average_loss

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = r"NewVAECode/configs/edges_median.json"
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    cvae_model = CVAECfg(
        nfft=cfg['hrtf']['nfft'],
        cfg={
            'labels': cfg['hrtf']['labels'],
            'encoder_layer_sizes': cfg['hrtf']['encoder_layer_sizes'],
            'decoder_layer_sizes': cfg['hrtf']['decoder_layer_sizes'],
            'latent_size': cfg['hrtf']['latent_size'],
            }
    ).to(device)
    cvae_path= r"weights/version_13/checkpoints/epoch=128-step=384677.ckpt"
    cvae_checkpoint = torch.load(cvae_path, map_location=device)
    cvae_state_dict = cvae_checkpoint['state_dict']
    cvae_model.load_state_dict(cvae_state_dict)
    cvae_model = cvae_model.to(device)
    cvae_model.eval()

    vae_model = VAECfg(
            input_size=[cfg['ears']['img_size'], cfg['ears']['img_size']],
            cfg={
                'input_channels': cfg['ears']['img_channels'],
                'encoder_channels': cfg['ears']['encoder_channels'],
                'latent_size': cfg['ears']['latent_size'],
                'decoder_channels': cfg['ears']['decoder_channels'],
                'kl_coeff': cfg['ears']['kl_coeff'],
                'use_inception': cfg['ears']['use_inception'],
                'repeat_per_block': cfg['ears']['repeat_per_block']
            }
        ).to(device)
    vae_path= r"weights/version_17/checkpoints/epoch=925-step=35188.ckpt"
    vae_checkpoint = torch.load(vae_path, map_location=device)
    vae_state_dict = vae_checkpoint['state_dict']
    vae_model.load_state_dict(vae_state_dict)
    vae_model = vae_model.to(device)
    vae_model.eval()        

    dnn_model = DNNCfg(
        cfg={
            'labels': cfg['latent']['labels'],
            'z_ears_size': cfg['latent']['z_ears_size'],
            'z_hrtf_size': cfg['latent']['z_hrtf_size'],
            'hidden_layers': cfg['latent']['hidden_layers'],
            'dropout_rate': cfg['latent']['dropout_rate'],
        }
    ).to(device)
    dnn_path= r"weights/version_2/checkpoints/epoch=44-step=134189.ckpt"
    dnn_checkpoint = torch.load(dnn_path, map_location=device)
    dnn_state_dict = dnn_checkpoint['state_dict']
    dnn_model.load_state_dict(dnn_state_dict)
    dnn_model = dnn_model.to(device)
    dnn_model.eval()      


    image_dir = "Ear_image_gray"
    hrtf_dir = "FFT_HRTF"
    dataset_paths = split_dataset(image_dir, hrtf_dir)
    
    data_transform = transforms.Compose([
        transforms.Resize(cfg['ears']['img_size']),
        transforms.ToTensor(),
        transforms.Grayscale(cfg['ears']['img_channels']),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = SonicomDataSetLSD(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        transform=data_transform,
        calc_mean=True,
        status="cvae",
        mode="left"
    )
    
    test_dataset = SonicomDataSetLSD(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        transform=data_transform,
        calc_mean=False,
        status="cvae",
        mode="left",
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=50,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    LSD = calculate_lsd(test_loader, cvae_model, vae_model, dnn_model, device)
    print("LSD", LSD)

if __name__ == '__main__':
    main()
