import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet,SonicomDataSetLeft
from vae_incept_cfg import InceptionVAECfg as VAECfg  
from utils import split_dataset, train_one_epoch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm
import sys

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
            'kl_coeff': cfg['ears']['kl_coeff'],
            'use_inception': cfg['ears']['use_inception'],
            'repeat_per_block': cfg['ears']['repeat_per_block']
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

    # 创建数据集
    train_dataset = SonicomDataSetLeft(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        device=device,
        transform=data_transform,
        calc_mean=True,
        mode="left"
    )
    
    test_dataset = SonicomDataSetLeft(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        device=device,
        transform=data_transform,
        calc_mean=False,
        status="test",
        mode="left",
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=18,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    print(train_dataset[0])
    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # 训练循环
    num_epochs = 480*5
    '''
    optimizers, lr_schedulers = model.configure_optimizers()
    optimizer = optimizers[0]
    lr_scheduler = lr_schedulers[0]

    
    for epoch in range(0, num_epochs):
        # 训练
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        model.training_epoch_end()
    '''    
    # 初始化 logger
    logger = TensorBoardLogger("tb_logs", name="vae_5.8_model")

    # 创建 Trainer 实例并传递 logger
    trainer = Trainer(max_epochs=num_epochs, logger=logger)

    # 开始训练
    trainer.fit(model, train_loader,test_loader)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新增配置文件参数
    parser.add_argument('--cfg-path', type=str, help='Path to model config file',default= 'NewVAECode/configs/edges_median.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-name', default='vae_conv', help='Output model name')
    parser.add_argument('--device', default='cuda:0', help='Device id')
    
    opt = parser.parse_args()
    main(opt)
