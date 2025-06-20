import os
import argparse
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet,SonicomDataSetDNN
from dnn_cfg import DNNCfg
from vae_incept_cfg import InceptionVAECfg as VAECfg  
from cvae_dense_cfg import CVAECfg 
from utils import split_dataset, train_one_epoch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    with open(args.cfg_path, 'r') as f:
        cfg = json.load(f)

    model = DNNCfg(
        cfg={
            'labels': cfg['latent']['labels'],
            'z_ears_size': cfg['latent']['z_ears_size'],
            'z_hrtf_size': cfg['latent']['z_hrtf_size'],
            'hidden_layers': cfg['latent']['hidden_layers'],
            'dropout_rate': cfg['latent']['dropout_rate'],
        }
    ).to(device)
    

    # 数据集准备（保持原有逻辑）
    image_dir = "Ear_image_gray_Wi"
    hrtf_dir = "FFT_HRTF_Wi"
    dataset_paths = split_dataset(image_dir, hrtf_dir)
    
    # 数据转换（保持通道数一致）
    data_transform = transforms.Compose([
        transforms.Resize(cfg['ears']['img_size']),
        transforms.ToTensor(),
        transforms.Grayscale(cfg['ears']['img_channels']),
        transforms.Normalize([0.5], [0.5])
    ])

    # 创建数据集
    train_dataset = SonicomDataSetDNN(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        transform=data_transform,
        calc_mean=True,
        status="cvae",
        mode="left"
    )
    
    test_dataset = SonicomDataSetDNN(
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
    

    batch_train_test = next(iter(train_loader))
    print("Batch keys:", batch_train_test.keys())  
    print("Shape of z_ears:", batch_train_test["z_ears"].shape) 
    print("Shape of z_hrtf:", batch_train_test["z_hrtf"].shape)
    #print("Shape of position:", batch_train_test["position"].shape)     
    # 输出为: Shape of z_ears: torch.Size([50, 1, 256]) Shape of z_hrtf: torch.Size([50, 1,32])
    #         Shape of position: torch.Size([50, 1,3])

    
    test_loader = DataLoader(
        test_dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )

    #batch_example = next(iter(test_loader))
    #model.example_input_array = batch_example["left_image"]

    # 训练循环
    num_epochs = 480*5

    # 初始化 logger
    logger = TensorBoardLogger("tb_logs", name="dnn_6.17_model")

    trainer = Trainer(
        max_epochs=num_epochs,
        logger=logger,
        val_check_interval=1.0,  # 确保验证只在每个 epoch 结束后进行
    )

    # 开始训练
    trainer.fit(model, train_loader,test_loader)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 新增配置文件参数
    parser.add_argument('--cfg-path', type=str, help='Path to model config file',default= 'NewVAECode/configs/edges_widespread.json')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--model-name', default='vae_conv', help='Output model name')
    parser.add_argument('--device', default='cuda:0', help='Device id')
    
    opt = parser.parse_args()
    main(opt)
