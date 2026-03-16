"""
实现联合训练卷积神经网络（CNN）和预训练的自编码器（VQVAE）来处理HRTF数据。
CNN的输出和VQVAE的解码器（Decoder）连接，以实现端到端的训练。
在本代码中，VQVAE没有使用VQ进行量化，目的是和有VQ的效果进行对比。
"""
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from TestNet import ResNet3DClassifier as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet, OnlyHRTFDataSet
from utils import split_dataset, train_one_epoch, evaluate
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from AE import HRTF_VQVAE
from AEconfig import transformer_encoder_settings, transformer_decoder_settings, encoder_out_vec_num, \
    embed_dim, num_codebook_embeddings, use_VQ, input_pos_as_seq, \
        tolerance_for_calc_threshold, decay
import time

def main():
    # 设备配置
    CNN_class = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    weightname = "best_model_1111-0055.pth"
    # weightname = "no_pretrain"
    VQVAE_path = "AE_related/HRTF_VQVAE/savetime_10-26_06-35.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = False  # 是否使用差值HRTF数据

    if CNN_class == "3DResNet":
        weightdir = "AE_related/CNN3D"
        ear_dir = "Ear_voxel_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        # positions_chosen_num = 793
        model = threeDResnet(d_model=embed_dim, encoder_out_vec_num=encoder_out_vec_num).to(device)
        inputform = "voxel"
    elif CNN_class == "2DResNet":
        weightdir = "AE_related/CNN"
        ear_dir = "Ear_image_gray_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        # positions_chosen_num = 793
        model = twoDResnet(d_model=embed_dim, encoder_out_vec_num=encoder_out_vec_num).to(device)
        inputform = "image"


    if os.path.exists(modelpath):
        print("Load model from", modelpath)
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    
    if VQVAE_path.endswith(".pth"):
        state_dict = torch.load(VQVAE_path, map_location=device,weights_only=True)
    else:
        state_dict = torch.load(VQVAE_path, map_location=device,weights_only=True)['model_state_dict']
    hrtf_encoder = HRTF_VQVAE(
        hrtf_row_len=state_dict['encoder.input_projection.weight'].shape[1],
        encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
        embed_dim=state_dict['encoder.input_projection.weight'].shape[0],
        encoder_transformer_config=transformer_encoder_settings,
        decoder_transformer_config=transformer_decoder_settings,
        num_embeddings=num_codebook_embeddings,
        use_VQ=use_VQ,
        input_pos_as_seq=input_pos_as_seq,
        tolerance_for_calc_threshold=tolerance_for_calc_threshold,
        decay=decay
    ).to(device)
    hrtf_encoder.load_state_dict(state_dict)
    print("Load HRTF encoder from", VQVAE_path)
    
    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)

    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=usediff,
        calc_mean=usediff,
        inputform=inputform,
        mode="right",
        provided_feature=None
    )

    
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=inputform,
        mode="right",
        use_diff=usediff,                  
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right,
        provided_feature=None
    )
    # 创建数据加载器
    batch_size = 12
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )

    auxiliary_loader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=6,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    # 优化器参数
    base_lr = 2e-4
    backbone_coeff = 0.1
    decoder_coeff = 0.5
    weight_decay = 1e-2
    pg0, pg1 = [], []
    pg2, pg3 = [], []  # optimizer parameter groups，对应关系：
    # CNN model: pg0: weight decay, pg1: no weight decay
    # CNN model head: pg2: weight decay, pg3: no weight decay
    for k, v in model.named_parameters():
        if 'vq_layer' in k:
            if 'bias' in k or 'bn' in k or 'downsample.1' in k:
                pg3.append(v)
            else:
                pg2.append(v)
        else:
            if 'bias' in k or 'bn' in k or 'downsample.1' in k:
                pg1.append(v)
            else:
                pg0.append(v)
    pg4, pg5 = [], []  # optimizer parameter groups for VQVAE
    # VQVAE: pg4: weight decay, pg5: no weight decay
    for k, v in hrtf_encoder.named_parameters():
        if 'bias' in k or 'bn' in k or 'norm' in k:
            pg5.append(v)
            # print("no weight decay:\t", k)
        else:
            pg4.append(v)
            # print("weight decay:\t", k)

    optimizer = optim.AdamW(pg2, lr=base_lr, weight_decay=weight_decay)
    optimizer.add_param_group({'params': pg3, 'weight_decay': 0.0})

    optimizer.add_param_group({'params': pg0, "lr": base_lr*backbone_coeff, 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': pg1, "lr": base_lr*backbone_coeff, 'weight_decay': 0.0})
    optimizer.add_param_group({'params': pg4, "lr": base_lr*decoder_coeff, 'weight_decay': weight_decay})
    optimizer.add_param_group({'params': pg5, "lr": base_lr*decoder_coeff, 'weight_decay': 0.0})
    del pg0, pg1, pg2, pg3, pg4, pg5  # free memory
    # 学习率调度器: 余弦退火调度器
    num_epochs = 60
    warmsteps = int(0.05 * num_epochs)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmsteps, num_training_steps=num_epochs)
    # 训练循环
    log_dir = "AE_related/joint_training/without_VQ/logs"
    CNN_save_path = "AE_related/joint_training/without_VQ" + (f"/CNN3D" if CNN_class=="3DResNet" else f"/CNN2D")
    Decoder_save_path = "AE_related/joint_training/without_VQ/HRTF_VQVAE"
    if os.path.exists(CNN_save_path) is False:
        os.makedirs(CNN_save_path)
    if os.path.exists(Decoder_save_path) is False:
        os.makedirs(Decoder_save_path)
    timestamp = time.strftime('%m%d-%H%M')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)+sum(p.numel() for p in hrtf_encoder.parameters() if p.requires_grad)
    config_params = {
    "use_diff": usediff,
    "batch_size": batch_size,
    "used_CNN": modelpath,
    "used_VQVAE": VQVAE_path,
    "width_per_hrtf_row": state_dict['encoder.input_projection.weight'].shape[1],
    "encoder_out_vec_num": encoder_out_vec_num,
    "embed_dim": embed_dim,
    "num_codebook_embeddings": num_codebook_embeddings,
    **{f"encoder_transformer/{k}": v for k, v in transformer_encoder_settings.items()}, # 平铺字典
    **{f"decoder_transformer/{k}": v for k, v in transformer_decoder_settings.items()}, # 平铺字典
    "Total Parameters": total_params,
    "CNN_output": "Embeddings, but no use of VQ"
    }
    config_text = "## Model Architecture Configuration\n\n"
    config_text += "| Parameter | Value |\n"
    config_text += "|:---|:---|\n"
    for key, value in config_params.items():
        config_text += f"| {key} | {value} |\n"
    writer = SummaryWriter(log_dir=f"{log_dir}/VQVAE_{CNN_class}{timestamp}")
    writer.add_text("model_config", config_text)
    print(f"Total parameters: {total_params}")
    reconstruction_loss_fn = torch.nn.MSELoss()
    for epoch in range(0, num_epochs + 1):
        # 训练
        model.train()
        hrtf_encoder.train()
        epoch_loss_recon = 0
        epoch_loss_vq = 0
        vq_loss = torch.zeros([1]).cuda()
        train_progress_bar = tqdm(train_loader, file=sys.stdout)
        for i, batch in enumerate(train_progress_bar):
            optimizer.zero_grad()
            hrtf = batch["hrtf"].to(device) # 假设形状是 (batch, 793, 108)
            pos = batch["position"].to(device)   # (batch, 793, 3)
            right_picture = batch["right_voxel"].to(device)

            pred_z = model(right_picture, device=device)
            reconstructed_hrtf = hrtf_encoder.decoder(pred_z, pos)
            total_loss = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
            total_loss.backward()
            optimizer.step()

            epoch_loss_recon += total_loss.item()
            epoch_loss_vq += vq_loss.item()
            
            train_progress_bar.desc = (f"[train epoch {epoch+1}] loss_recon: {epoch_loss_recon/(i+1):.2f} "
                                    f"loss_vq: {epoch_loss_vq/(i+1):.2f} "
                                    f"lr: {optimizer.param_groups[0]['lr']:.2e} ")

        avg_recon_loss_train = epoch_loss_recon / len(train_loader)
        avg_vq_loss_train = epoch_loss_vq / len(train_loader)
        writer.add_scalar("Loss/train", avg_recon_loss_train, epoch)
        writer.add_scalar("Loss/train_vq", avg_vq_loss_train, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        # 验证
        model.eval()
        hrtf_encoder.eval()
        val_loss_recon = 0
        val_loss_vq = 0
        vq_loss_val = torch.zeros([1]).cuda()
        with torch.no_grad():
            val_progress_bar = tqdm(test_loader, file=sys.stdout)
            for step, batch in enumerate(val_progress_bar):
                hrtf = batch["hrtf"].to(device) # 假设形状是 (batch, 793, 108)
                pos = batch["position"].to(device)   # (batch, 793, 3)
                right_picture = batch["right_voxel"].to(device)
                
                pred_z = model(right_picture, device=device)
                reconstructed_hrtf = hrtf_encoder.decoder(pred_z, pos)
                recon_loss_val = reconstruction_loss_fn(reconstructed_hrtf, hrtf)
                val_loss_recon += recon_loss_val.item()
                val_loss_vq += vq_loss_val.item()
                val_progress_bar.desc = (f"[valid epoch {epoch+1}] loss_recon: {val_loss_recon/(step+1):.3f} "
                                        f"loss_vq: {val_loss_vq/(step+1):.3f}")
        avg_recon_loss_val = val_loss_recon / len(test_loader)
        avg_vq_loss_val = val_loss_vq / len(test_loader)
        writer.add_scalar("Loss/val", avg_recon_loss_val, epoch)
        writer.add_scalar("Loss/val_vq", avg_vq_loss_val, epoch)

        # 更新学习率调度器
        scheduler.step() # 在每个 epoch 结束后（或验证后）调用

        # 保存模型
        if (epoch + 1) % 30 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': hrtf_encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(checkpoint, f"{Decoder_save_path}/savetime_{timestamp}.pt")
            torch.save(model.state_dict(), f"{CNN_save_path}/best_model_{timestamp}.pth")
            print(f"Checkpoint saved at epoch {epoch+1}")


if __name__ == "__main__":
    main()