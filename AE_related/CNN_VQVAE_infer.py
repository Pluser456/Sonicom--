import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TestNet import TestNet as threeDResnetANP
from TestNet import ResNet3D as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet
from utils import split_dataset
from tqdm import tqdm

from AE import HRTF_VQVAE
from AEconfig import pos_dim_for_each_row, \
    num_hrtf_rows, width_per_hrtf_row, transformer_encoder_settings, decoder_mlp_layers, encoder_out_vec_num, \
    num_codebook_embeddings, commitment_cost_beta

def main():
    # 设备配置
    current_model = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    weightname = "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = True  # 是否使用差值HRTF数据

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
        ear_dir = "Ear_image_gray_Wi"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        # positions_chosen_num = 793
        model = twoDResnet().to(device)
        inputform = "image"

    hrtf_encoder = HRTF_VQVAE(
        hrtf_row_width=width_per_hrtf_row,
        hrtf_num_rows=num_hrtf_rows,
        encoder_out_vec_num=encoder_out_vec_num, # 编码器输出序列长度
        encoder_transformer_config=transformer_encoder_settings,
        num_embeddings=num_codebook_embeddings,
        commitment_cost=commitment_cost_beta,
        pos_dim_per_row=pos_dim_for_each_row,
        decoder_mlp_hidden_dims=decoder_mlp_layers
    ).to(device)

    if os.path.exists(modelpath):
        print("Load model from", modelpath)
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    hrtf_encoder.load_state_dict(torch.load("HRTFAEweights/model-rvqvae-180.pth", map_location=device,weights_only=True))
    print("Load HRTF encoder from", "HRTFAEweights/model-rvqvae-180.pth")

    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)
    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=usediff,
        calc_mean=True,
        status="test", # 因为这里希望坐标是按顺序输入的
        inputform=inputform,
        mode="right"
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
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_dataset.collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    
    # log_dir = f"runs/{current_model}"
    # writer = SummaryWriter(log_dir=f"{log_dir}/VQVAE_{time.strftime('%m%d-%H%M')}")
    #     # 验证
    # val_loss = evaluate(model, test_loader, device, epoch=epoch, auxiliary_loader=auxiliary_loader)
    model.eval()
    hrtf_encoder.eval()
    with torch.no_grad():
        criterion = nn.MSELoss()
        progressbar = tqdm(train_loader)
        total_loss = 0
        size = 0
        for i, batch in enumerate(progressbar):
            hrtf = batch["hrtf"].to(device).unsqueeze(1)
            pos = batch["position"].to(device)
            right_picture = batch["right_voxel"].to(device)
            pred, _ = model(right_picture, device=device) # [batch_size, 18]
            pred = pred.reshape(-1, 3, 3)
            # pred = pred.permute(1, 0, 2, 3) # [2, batch_size, 3, 3]
            # pred =torch.randint_like(pred, low=0, high=num_codebook_embeddings) # 随机生成索引以测试
            zq = hrtf_encoder.vq_layer.get_output_from_indices(pred)
            output = hrtf_encoder.decoder(zq, pos)
            loss = criterion(output, hrtf)
            total_loss += loss.item() * hrtf.shape[0]
            size += hrtf.shape[0]
            progressbar.desc = f"Test Loss: {total_loss / size:.3f}"


if __name__ == "__main__":
    main()