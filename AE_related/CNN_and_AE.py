import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from TestNet import TestNet as threeDResnetANP
from TestNet import ResNet3D as threeDResnet
from TestNet import ResNet2D as twoDResnet
from new_dataset import SonicomDataSet, OnlyHRTFDataSet
from utils import split_dataset, train_one_epoch, evaluate
from tqdm import tqdm
import sys
from AE import HRTFAutoencoder
from AEconfig import latent_dim, pos_dim_for_each_row, num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers

def main():
    # 设备配置
    current_model = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    weightname = "mode.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = False  # 是否使用差值HRTF数据

    if current_model == "3DResNetANP":
        weightdir = "./ANP3Dweights"
        ear_dir = "Ear_voxel"
        isANP = True
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)

        # 从预训练模型加载权重
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793 # 训练集每个文件选择的方位数
        model = threeDResnetANP(target_num_anp=5, positions_num=positions_chosen_num).to(device)
        inputform ="voxel"
    elif current_model == "3DResNet":
        weightdir = "./CNN3Dweights"
        ear_dir = "Ear_voxel"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793
        model = threeDResnet().to(device)
        inputform = "voxel"
    elif current_model == "2DResNet":
        weightdir = "./CNNweights"
        ear_dir = "Ear_image_gray"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        positions_chosen_num = 793
        model = twoDResnet().to(device)
        inputform = "image"


    if os.path.exists(modelpath):
        print("Load model from", modelpath)
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    
    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF",inputform=inputform)
    
    hrtf_feature = get_hrtf_feature(dataset_paths["train_hrtf_list"], 
                                    use_diff=usediff,
                                    calc_mean=True,
                                    status="test",
                                    mode="left")

    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        positions_chosen_num=positions_chosen_num,
        use_diff=usediff,
        calc_mean=True,
        inputform=inputform,
        mode="left"
    )
    
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=inputform,
        mode="left",
        use_diff=usediff,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
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
        batch_size=1,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    # 训练循环
    num_epochs = 480*5
    best_loss = 300
    
    patience = 10  # 早停的容忍次数
    patience_counter = 0

    for epoch in range(0, num_epochs + 1):
        # 训练
        train_one_epoch(model, optimizer, train_loader, device, epoch)

        # 验证
        train_dataset.turn_auxiliary_mode(True)
        val_loss = evaluate(model, test_loader, device, epoch, auxiliary_loader=auxiliary_loader)
        train_dataset.turn_auxiliary_mode(False)

        # 检查是否是最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), f"{weightdir}/best_model.pth")
            print(f"Saved best model with validation loss: {best_loss:.4f}")
        else:
            patience_counter += 1

        # 检查早停条件
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs with best validation loss: {best_loss:.4f}")
            break

        # 保存当前模型
        if epoch % 50 == 0:
            torch.save(model.state_dict(), f"{weightdir}/model-{epoch}.pth")
            print(f"Saved model at epoch {epoch}")

def get_hrtf_feature(hrtf_files, 
                 status="train",
                 calc_mean=True, use_diff=True,
                 mode="both", provided_mean_left=None, provided_mean_right=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hrtf_encoder = HRTFAutoencoder(
            latent_feature_dim=latent_dim,
            pos_dim_per_row=pos_dim_for_each_row,
            hrtf_num_rows=num_hrtf_rows,
            hrtf_row_width=width_per_hrtf_row,
            decoder_mlp_hidden_dims=decoder_mlp_layers,
            encoder_transformer_config=transformer_encoder_settings
        ).to(device)
    hrtf_encoder.load_state_dict(torch.load("HRTFAEweights/model-800.pth", map_location=device,weights_only=True))
    dataset = OnlyHRTFDataSet(hrtf_files, status=status, calc_mean=calc_mean, use_diff=use_diff, mode=mode, provided_mean_left=provided_mean_left, provided_mean_right=provided_mean_right)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    hrtf_data = []
    hrtf_encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, file=sys.stdout):
            hrtf = batch["hrtf"].to(device).unsqueeze(1)
            hrtf_feature = hrtf_encoder.encoder(hrtf)
            hrtf_data.append(hrtf_feature)
    hrtf_data = torch.cat(hrtf_data, dim=0)
    return hrtf_data
            

if __name__ == "__main__":
    main()