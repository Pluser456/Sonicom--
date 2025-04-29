import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from TestNet import TestNet
from new_dataset import SonicomDataSet
from utils import split_dataset, train_one_epoch, evaluate

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists("./ANPweights") is False:
        os.makedirs("./ANPweights")

            # 从预训练模型加载权重
    modelpath = "ANPweights/model-2400.pth"
    model = TestNet().to(device)
    if os.path.exists(modelpath):
        print("Load model from", modelpath)

        model.load_state_dict(torch.load(modelpath, map_location=device, weights_only=True))
    
    # 数据转换
    data_transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 数据分割
    dataset_paths = split_dataset("Ear_image_gray", "FFT_HRTF")
    
    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        device=device,
        transform=data_transform,
        calc_mean=True,
        mode="left"
    )
    
    test_dataset = SonicomDataSet(
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
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
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
        batch_size=50,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 480*5
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        if epoch % 150 == 0:
            # 验证
            train_dataset.turn_auxiliary_mode(True)
            val_loss = evaluate(model, test_loader, device, epoch, auxiliary_loader=auxiliary_loader)
            train_dataset.turn_auxiliary_mode(False)
            # 保存最佳模型
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), "./ANPweights/best_model.pth")
                print(f"Saved best model with validation loss: {best_loss:.4f}")
            torch.save(model.state_dict(), "./ANPweights/model-{}.pth".format(epoch))
            print(f"Saved model at epoch {epoch}")

if __name__ == "__main__":
    main()