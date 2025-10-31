import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from TestNet import ResNet3DClassifier as threeDResnet
from TestNet import ResNet2DClassifier as twoDResnet
from new_dataset import SonicomDataSet
from utils import split_dataset
from tqdm import tqdm

from AE import HRTF_VQVAE
from AEconfig import transformer_encoder_settings, transformer_decoder_settings, encoder_out_vec_num, \
    hrtf_row_len, num_codebook_embeddings, commitment_cost_beta, embed_dim, use_VQ, input_pos_as_seq, \
        tolerance_for_calc_threshold, decay

def main():
    # 设备配置
    current_model = "2DResNet" # ["3DResNetANP", "3DResNet", "2DResNetANP", "2DResNet"]
    VQVAE_path = "AE_related/HRTF_VQVAE/savetime_10-26_20-49.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usediff = False  # 是否使用差值HRTF数据

    if current_model == "3DResNet":
        weightname = "best_model_1031-0215.pth"
        weightdir = "AE_related/CNN3D"
        ear_dir = "Ear_voxel_Wi"
        isANP = False
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        model = threeDResnet(d_model=embed_dim, encoder_out_vec_num=encoder_out_vec_num).to(device)
        inputform = "voxel"
    elif current_model == "2DResNet":
        weightname = "best_model_1030-1756.pth"
        weightdir = "AE_related/CNN"
        ear_dir = "Ear_image_gray_Wi"
        if os.path.exists(weightdir) is False:
            os.makedirs(weightdir)
        modelpath = f"{weightdir}/{weightname}"
        model = twoDResnet(d_model=embed_dim, encoder_out_vec_num=encoder_out_vec_num).to(device)
        inputform = "image"

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
    print("Load HRTF encoder")

    if os.path.exists(modelpath):
        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
        print("Load model from", modelpath)

    # 数据分割
    dataset_paths = split_dataset(ear_dir, "FFT_HRTF_Wi",inputform=inputform)
    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=usediff,
        calc_mean=usediff,
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
        progressbar = tqdm(test_loader)
        total_loss = 0
        total_acc = 0
        size = 0
        for i, batch in enumerate(progressbar):
            hrtf = batch["hrtf"].to(device)
            pos = batch["position"].to(device)
            right_picture = batch["right_voxel"].to(device)
            pred = model(right_picture, device=device)
            with torch.no_grad():
                zq, idx, _ = hrtf_encoder.quantize(pred)
            # pred = pred.reshape(-1, 2, 3, 3)
            # pred = pred.permute(1, 0, 2, 3) # [2, batch_size, 3, 3]
            # pred =torch.randint_like(pred, low=0, high=num_codebook_embeddings) # 随机生成索引以测试

            _, _, true_pred = hrtf_encoder(hrtf, pos)

            output = hrtf_encoder.decoder(zq, pos)
            loss = criterion(output, hrtf)
            acc = (idx == true_pred).float().mean()
            total_loss += loss.item() * hrtf.shape[0]
            total_acc += acc.item() * hrtf.shape[0]
            size += hrtf.shape[0]
            progressbar.desc = f"Loss: {total_loss / size:.3f}, Acc: {total_acc / size:.3f}"


if __name__ == "__main__":
    main()