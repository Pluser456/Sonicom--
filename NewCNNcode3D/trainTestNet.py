import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet
from torch.utils.data import DataLoader
from TestNet import TestNet as create_model
from utils import split_dataset, train_one_epoch, evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model().to(device)
    positions_chosen_num = 793 # 训练集每个文件选择的方位数

    if os.path.exists("./CNNweights") is False:
        os.makedirs("./CNNweights")


    dataset_paths = split_dataset("Ear_voxel", "FFT_HRTF")
   # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        device=device,
        positions_chosen_num=positions_chosen_num,
        calc_mean=True,
        mode="left"
    )
    
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        device=device,
        calc_mean=False,
        status="test",
        mode="left",
        provided_mean_left=train_dataset.log_mean_hrtf_left,
        provided_mean_right=train_dataset.log_mean_hrtf_right
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn
    )
    
    

    # auxiliary_loader = DataLoader(
    #     train_dataset,
    #     batch_size=len(train_dataset),
    #     shuffle=True,
    #     collate_fn=train_dataset.collate_fn
    # )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=40,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
  
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    
    # 训练循环
    num_epochs = 480*5
    best_loss = 23.9
    
    for epoch in range(0, num_epochs + 1):
        # 训练
        # train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        if epoch % 50 == 0:
            # 验证
            # train_dataset.turn_auxiliary_mode(True)
            val_loss = evaluate(model, test_loader, device, epoch)
            # train_dataset.turn_auxiliary_mode(False)
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "./ANP3Dweights/best_model.pth")
                print(f"Saved best model with validation loss: {best_loss:.4f}")
            torch.save(model.state_dict(), "./ANP3Dweights/model-{}.pth".format(epoch))
            print(f"Saved model at epoch {epoch}")

if __name__ == "__main__":
    main()