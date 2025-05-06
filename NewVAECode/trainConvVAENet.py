import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet
from vae_conv_cfg import VAECfg  # 直接使用VAE配置类
from utils import split_dataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录
    os.makedirs("./VAEweights", exist_ok=True)
    tb_writer = SummaryWriter()

    # 加载配置文件（参考ear_to_prtf的逻辑）
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # 初始化VAE模型（核心修改点）
    model = VAECfg(
        input_size=[cfg['ears']['img_size'], cfg['ears']['img_size']],
        cfg={
            'input_channels': cfg['ears']['img_channels'],
            'encoder_channels': cfg['ears']['encoder_channels'],
            'latent_size': cfg['ears']['latent_size'],
            'decoder_channels': cfg['ears']['decoder_channels'],
            'kl_coeff': cfg['ears']['kl_coeff']
        }
    ).to(device)

    # 数据集准备（保持原有逻辑）
    image_dir = "Ear_image_gray"
    hrtf_dir = "FFT_HRTF"
    dataset_paths = split_dataset(image_dir, hrtf_dir)
    
    # 数据转换（保持通道数一致）
    data_transform = transforms.Compose([
        transforms.Resize(cfg['ears']['img_size']),
        transforms.ToTensor(),
        transforms.Grayscale(cfg['ears']['img_channels']),
        transforms.Normalize([0.5], [0.5])
    ])

    # 实例化数据集（修改目标为图像本身）
    train_dataset = SonicomDataSet(
        hrtf_files=dataset_paths['train_hrtf_list'],
        left_images=dataset_paths['left_train'],
        right_images=dataset_paths['right_train'],
        transform=data_transform,
        mode="left",
        target_type="image"  # 假设数据集支持返回图像本身作为目标
    )

    val_dataset = SonicomDataSet(
        hrtf_files=dataset_paths['test_hrtf_list'],
        left_images=dataset_paths['left_test'],
        right_images=dataset_paths['right_test'],
        transform=data_transform,
        mode="left",
        target_type="image",
        calc_mean=False,
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn
    )

    # 使用Lightning Trainer（简化训练流程）
    logger = TensorBoardLogger("tb_logs", name="vae_conv")
    trainer = Trainer(
        max_epochs=args.epochs,
        logger=logger,
        devices=1 if str(device) == 'cuda:0' else 0,
        accelerator='gpu' if 'cuda' in str(device) else 'cpu',
        enable_checkpointing=False
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 保存最终模型
    torch.save(model.state_dict(), f"./VAEweights/{args.model_name}_final.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新增配置文件参数
    parser.add_argument('--cfg-path', type=str, required=True, help='Path to model config file')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-name', default='vae_conv', help='Output model name')
    parser.add_argument('--device', default='cuda:0', help='Device id')
    
    opt = parser.parse_args()
    main(opt)
