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
from src.dataset.hrtf import SonicomDataSetLSD
from src.utils.data import split_dataset, train_one_epoch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

def calculate_lsd(dataloader, cvae_model, vae_model, dnn_model, device):
    #LSD = 3.62264050
    #LSD = 3.8496014293173686
    pred_single_hrtf_mat =[]
    pred_hrtf_tensor = []
    single_hrtf_mat =[]
    hrtf_tensor = []
    lsd_list = []
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
            pred_single_hrtf_mat.append(hrtf_reconstructed)
            single_hrtf_mat.append(hrtf)
            if (i+1) % 2562 == 0:
                single_hrtf_mat = torch.cat(single_hrtf_mat, dim=0)
                pred_single_hrtf_mat = torch.cat(pred_single_hrtf_mat, dim=0)
                hrtf_tensor.append(single_hrtf_mat)
                pred_hrtf_tensor.append(pred_single_hrtf_mat)
                single_lsd = torch.sqrt(mse_loss(torch.abs(single_hrtf_mat), torch.abs(pred_single_hrtf_mat)))
                lsd_list.append(single_lsd.item())
                print(f"\nSubject {(i+1)//2562}: LSD = {single_lsd.item():.6f}")
                pred_single_hrtf_mat =[]
                single_hrtf_mat =[]
            # loss = mse_loss(torch.abs(hrtf), torch.abs(hrtf_reconstructed))
            # if loss.is_cuda:
            #     loss = loss.cpu()

            # total_loss += loss.item()  # 累加损失
            # print(f"Batch {i + 1}: Loss = {lsd_loss.item():.4f}, Total Loss = {total_loss / (i + 1):.4f}")

            '''
            # 打印原始hrtf曲线
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(hrtf[0, :].cpu().numpy(), label='Original HRTF')
            plt.title('Original HRTF Curve')
            plt.legend()

            # 打印重构的hrtf曲线
            plt.subplot(1, 2, 2)
            plt.plot(hrtf_reconstructed[0 ,:].cpu().numpy(), label='Reconstructed HRTF')
            plt.title('Reconstructed HRTF Curve')
            plt.legend()

            plt.show()
            break  # 处理完第一个batch后退出循环
            '''

    # 计算平均损失
    # average_loss = total_loss / len(dataloader)
    # print(f"Average MSE Loss: {average_loss}")
    average_lsd = sum(lsd_list) / len(lsd_list)
    # 计算逐频率点的LSD
    hrtf_tensor = torch.stack(hrtf_tensor, dim=0)
    pred_hrtf_tensor = torch.stack(pred_hrtf_tensor, dim=0)
    LSDvec = torch.sqrt(torch.mean((pred_hrtf_tensor - hrtf_tensor)**2, dim=1))  # 计算每个频率点的LSD
    avg_lsd_per_freq = torch.mean(LSDvec, dim=0).tolist()  # 计算所有样本在每个频率点的平均LSD
    return average_lsd, avg_lsd_per_freq

def calculate_txt(dataloader, cvae_model, vae_model, dnn_model, device):
    idx_0_0 = 1956
    idx_0_90 = 11
    idx_0_80 = 414
    idx_90_0 = 199
    idx_20_54 = 924
    with torch.no_grad():  # 不需要计算梯度
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            print(f"Batch {i + 1}")
            if i == idx_0_0 or i == idx_0_90 or i == idx_0_80 or i == idx_90_0 or i == idx_20_54:
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
                # 打印原始hrtf曲线
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(abs(hrtf[0, :].cpu().numpy()), label='Original HRTF')
                plt.title('Original HRTF Curve')
                plt.legend()

                # 打印重构的hrtf曲线
                plt.subplot(1, 2, 2)
                plt.plot(abs(hrtf_reconstructed[0 ,:].cpu().numpy()), label='Reconstructed HRTF')
                plt.title('Reconstructed HRTF Curve')
                plt.legend()
                plt.show()
                if i == idx_0_0:
                    np.savetxt('hrtf_VAE_0_0.txt', hrtf_reconstructed[0, :].cpu().numpy(), fmt='%.1f', header='Frequency (Hz)')
                elif i == idx_0_90:
                    np.savetxt('hrtf_VAE_0_90.txt', hrtf_reconstructed[0, :].cpu().numpy(), fmt='%.1f', header='Frequency (Hz)')
                elif i == idx_0_80:
                    np.savetxt('hrtf_VAE_0_80.txt', hrtf_reconstructed[0, :].cpu().numpy(), fmt='%.1f', header='Frequency (Hz)')
                elif i == idx_90_0:
                    np.savetxt('hrtf_VAE_90_0.txt', hrtf_reconstructed[0, :].cpu().numpy(), fmt='%.1f', header='Frequency (Hz)')
                elif i == idx_20_54:
                    np.savetxt('hrtf_VAE_20_54.txt', hrtf_reconstructed[0, :].cpu().numpy(), fmt='%.1f', header='Frequency (Hz)')

            if i >2562:
                break  


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg_path = r"NewVAECode/configs/edges_widespread.json"
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
    cvae_path= r"weights_ws2/version_8/checkpoints/epoch=3-step=203323.ckpt"
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
    vae_path= r"weights_ws2/version_1/checkpoints/epoch=23-step=4775.ckpt"
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
    dnn_path= r"weights_ws2/version_5/checkpoints/epoch=2-step=152492.ckpt"
    dnn_checkpoint = torch.load(dnn_path, map_location=device)
    dnn_state_dict = dnn_checkpoint['state_dict']
    dnn_model.load_state_dict(dnn_state_dict)
    dnn_model = dnn_model.to(device)
    dnn_model.eval()      


    image_dir = "Ear_image_gray_Wi"
    hrtf_dir = "FFT_HRTF_Wi"
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
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    LSD, avg_lsd_per_freq = calculate_lsd(test_loader, cvae_model, vae_model, dnn_model, device)
    print("LSD", LSD)
    path = f'HRTF可视化'
    input_type = '2D'
    np.savetxt(f'{path}\\lsd_VAE_{input_type}_Wi.txt', avg_lsd_per_freq, fmt='%.3f', header='LSD (dB)')

    # calculate_txt(test_loader, cvae_model, vae_model, dnn_model, device)

if __name__ == '__main__':
    main()
