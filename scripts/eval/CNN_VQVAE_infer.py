"""
CNN-VQVAE 推理评估脚本
用法:
    python scripts/eval/CNN_VQVAE_infer.py --config configs/eval/cnn-vqvae-eval.yaml
"""
import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.config import load_config
from src.models.TestNet import ResNet3DClassifier, ResNet2DClassifier
from src.models.AE import HRTF_VQVAE
from src.dataset.hrtf import SonicomDataSet
from src.utils.data import split_dataset


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CNN-VQVAE Evaluation')
    parser.add_argument('--config', type=str, default='configs/eval/cnn-vqvae-eval.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_pretrained_vqvae(config):
    """加载预训练的 VQVAE 模型"""
    device = torch.device(config.evaluation.device)

    # 加载 VQVAE 模型配置
    vqvae_config_path = config.pretrained.vqvae_config
    vqvae_config = load_config(vqvae_config_path)
    model_config = vqvae_config.model

    # 创建模型
    vqvae_model = HRTF_VQVAE(
        hrtf_row_len=model_config.hrtf_row_len,
        encoder_out_vec_num=model_config.encoder_out_vec_num,
        embed_dim=model_config.embed_dim,
        encoder_transformer_config=model_config.transformer_encoder_settings,
        decoder_transformer_config=model_config.transformer_decoder_settings,
        num_embeddings=model_config.codebook_size,
        use_VQ=model_config.use_VQ,
        input_pos_as_seq=model_config.input_pos_as_seq,
        decay=model_config.decay,
        tolerance_for_calc_threshold=model_config.tolerance_for_calc_threshold,
    ).to(device)

    # 加载权重
    vqvae_path = config.pretrained.vqvae_path
    if os.path.exists(vqvae_path):
        checkpoint = torch.load(vqvae_path, map_location=device, weights_only=False)
        vqvae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded VQVAE from {vqvae_path}")
    else:
        raise FileNotFoundError(f"VQVAE checkpoint not found: {vqvae_path}")

    vqvae_model.eval()
    return vqvae_model


def load_pretrained_cnn(config):
    """加载预训练的 CNN 模型"""
    device = torch.device(config.evaluation.device)

    # 加载 CNN 模型配置
    cnn_config_path = config.pretrained.cnn_config
    cnn_config = load_config(cnn_config_path)
    cnn_model_config = cnn_config.model

    # 创建模型
    model_type = config.cnn.model_type
    if model_type == "3DResNet":
        model = ResNet3DClassifier(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    elif model_type == "2DResNet":
        model = ResNet2DClassifier(
            num_classes=cnn_model_config.num_classes,
            encoder_out_vec_num=cnn_model_config.encoder_out_vec_num
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # 加载权重
    cnn_path = config.pretrained.cnn_path
    if os.path.exists(cnn_path):
        checkpoint = torch.load(cnn_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded CNN from {cnn_path}")
    else:
        raise FileNotFoundError(f"CNN checkpoint not found: {cnn_path}")

    model.eval()
    return model


def main():
    args = parse_args()

    # 加载配置
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        raise FileNotFoundError(f"Config file {args.config} not found.")

    # 设置设备和路径
    device = torch.device(config.evaluation.device)
    ear_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.ear_dir)
    hrtf_dir = str(Path(config.paths.data_dir) / config.dataset.name / config.dataset.hrtf_dir)

    # 加载 CNN 模型
    model = load_pretrained_cnn(config)

    # 加载 VQVAE 模型
    hrtf_encoder = load_pretrained_vqvae(config)

    # 分割数据集
    dataset_paths = split_dataset(
        ear_dir, hrtf_dir,
        inputform=config.dataset.input_form,
        n_folds=config.dataset.n_folds,
        val_fold=config.dataset.val_fold,
        seed=config.dataset.seed
    )

    # 创建训练数据集（用于计算 mean）
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        use_diff=config.dataset.use_diff,
        calc_mean=config.dataset.use_diff,
        status="test",
        inputform=config.dataset.input_form,
        mode=config.dataset.mode
    )

    # 创建测试数据集
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        calc_mean=False,
        status="test",
        inputform=config.dataset.input_form,
        mode=config.dataset.mode,
        use_diff=config.dataset.use_diff,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.evaluation.batch_size,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )



    # 评估
    model.eval()
    hrtf_encoder.eval()
    with torch.no_grad():
        criterion = nn.MSELoss()
        progressbar = tqdm(test_loader) if config.evaluation.eval_on == "test" else tqdm(train_loader)
        total_loss = 0
        total_acc = 0
        size = 0

        for batch in progressbar:
            hrtf = batch["hrtf"].to(device)
            pos = batch["position"].to(device)
            right_picture = batch["right_voxel"].to(device)
            pred, logits = model(right_picture, device=device) # (batch_size, encoder_out_vec_num)

            _, _, true_pred = hrtf_encoder(hrtf, pos)
            zq_list = []
            for i in range(hrtf_encoder.encoder_out_vec_num):
                zq_i = hrtf_encoder.vq_layer[i].get_output_from_indices(pred[:, i])
                zq_list.append(zq_i)
            zq = torch.stack(zq_list, dim=1)

            output = hrtf_encoder.decoder(zq, pos)
            loss = criterion(output, hrtf)
            acc = (pred == true_pred).float().mean()
            total_loss += loss.item() * hrtf.shape[0]
            total_acc += acc.item() * hrtf.shape[0]
            size += hrtf.shape[0]
            progressbar.desc = f"Loss: {total_loss / size:.3f}, Acc: {total_acc / size:.3f}"

    print(f"\nFinal - Loss: {total_loss / size:.3f}, Acc: {total_acc / size:.3f}")


if __name__ == "__main__":
    main()
