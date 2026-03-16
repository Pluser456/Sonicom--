import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from TestNet import ResNet3DClassifier as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet, OnlyHRTFDataSet
from utils import split_dataset, train_one_epoch, evaluate
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter
from AE import HRTF_VQVAE
from AEconfig import transformer_encoder_settings, transformer_decoder_settings, encoder_out_vec_num, \
    embed_dim, num_codebook_embeddings, use_VQ, input_pos_as_seq, \
        tolerance_for_calc_threshold, decay
import time

def main():
    # 设备配置
    current_model = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    weightname = "mode.pth"
    VQVAE_path = "AE_related/HRTF_VQVAE/savetime_10-27_22-09.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = False  # 是否使用差值HRTF数据

    if current_model == "3DResNet":
        weightdir = "AE_related/CNN3D"
        ear_dir = "Ear_voxel_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        # positions_chosen_num = 793
        model = threeDResnet(num_classes=num_codebook_embeddings, encoder_out_vec_num=encoder_out_vec_num).to(device)
        inputform = "voxel"
    elif current_model == "2DResNet":
        weightdir = "AE_related/CNN"
        ear_dir = "Ear_image_gray_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        # positions_chosen_num = 793
        model = twoDResnet(num_classes=num_codebook_embeddings, encoder_out_vec_num=encoder_out_vec_num).to(device)
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
    ).to(device).eval()
    hrtf_encoder.load_state_dict(state_dict)
    
    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)

    train_feature = get_hrtf_feature(dataset_paths["train_hrtf_list"], hrtf_encoder=hrtf_encoder, use_diff=usediff, status="test",mode="right")


    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=usediff,
        calc_mean=usediff,
        inputform=inputform,
        mode="right",
        provided_feature=train_feature
    )

    test_feature = get_hrtf_feature(dataset_paths["test_hrtf_list"], hrtf_encoder=hrtf_encoder, use_diff=usediff, status="test",mode="right", 
                                provided_mean_left=train_dataset.log_mean_hrtf_left,
                                provided_mean_right=train_dataset.log_mean_hrtf_right)
    
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
        provided_feature=test_feature
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
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
    optimizer = optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    # 学习率调度器: 每 step_size 个 epoch，学习率乘以 gamma
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.98) # 例如，每100个epoch学习率减半
    
    # 训练循环
    num_epochs = 480*5
    best_loss = 300
    best_acc = 0
    
    patience = 50  # 早停的容忍次数
    patience_counter = 0
    log_dir = weightdir
    timestamp = time.strftime('%m%d-%H%M')
    config_params = {
    "use_diff": usediff,
    "used_VQVAE": VQVAE_path,
    "decay": decay,
    "Total Parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "CNN_output": "Indexes of VQs"
    }
    config_text = "## Model Architecture Configuration\n\n"
    config_text += "| Parameter | Value |\n"
    config_text += "|:---|:---|\n"
    for key, value in config_params.items():
        config_text += f"| {key} | {value} |\n"
    writer = SummaryWriter(log_dir=f"{log_dir}/VQVAE_{current_model}{timestamp}")
    writer.add_text("model_config", config_text)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(0, num_epochs + 1):
        # 训练
        loss, acc = train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 验证
        val_loss, val_acc = evaluate(model, test_loader, device, epoch, auxiliary_loader=auxiliary_loader)
        writer.add_scalar("Loss/train", loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)
        # 更新学习率调度器
        scheduler.step() # 在每个 epoch 结束后（或验证后）调用

        # 检查是否是最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), f"{weightdir}/best_model_{timestamp}.pth")
            print(f"Saved best model with validation accuracy: {best_acc:.4f}")
        else:
            patience_counter += 1

        # 检查早停条件
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs with best validation accuracy: {best_acc:.4f}")
            break

        # 保存当前模型
        # if epoch % 50 == 0:
        #     torch.save(model.state_dict(), f"{weightdir}/model-{epoch}.pth")
        #     print(f"Saved model at epoch {epoch}")

def get_hrtf_feature(hrtf_files, hrtf_encoder, status="train", use_diff=True,
                 mode="both", provided_mean_left=None, provided_mean_right=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OnlyHRTFDataSet(hrtf_files, status=status, calc_mean=use_diff, use_diff=use_diff, mode=mode, provided_mean_left=provided_mean_left, provided_mean_right=provided_mean_right)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    hrtf_data = []
    hrtf_encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, file=sys.stdout):
            hrtf = batch["hrtf"].to(device)
            pos = batch["position"].to(device)
            _, _ ,idx = hrtf_encoder(hrtf, pos)
            hrtf_data.append(idx)
    hrtf_data = torch.cat(hrtf_data, dim=0)
    return hrtf_data

if __name__ == "__main__":
    main()