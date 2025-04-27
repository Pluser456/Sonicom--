import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from TestNet import TestNet, FeatureExtractor, PredictionNet
from new_dataset import SonicomDataSet, FeatureExtractorManager
from utils import split_dataset, train_one_epoch, evaluate

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # 数据分割
    dataset_paths = split_dataset("Ear_image_gray", "FFT_HRTF")
    
    # 初始化特征提取器管理器
    feature_manager = FeatureExtractorManager()
    
    # 创建数据集
    train_dataset = SonicomDataSet(
        dataset_paths["train_hrtf_list"],
        dataset_paths["left_train"],
        dataset_paths["right_train"],
        feature_extractor=feature_manager,
        transform=data_transform,
        calc_mean=True,
        mode="both"
    )
    
    test_dataset = SonicomDataSet(
        dataset_paths["test_hrtf_list"],
        dataset_paths["left_test"],
        dataset_paths["right_test"],
        feature_extractor=feature_manager,
        transform=data_transform,
        calc_mean=False,
        mode="both",
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=test_dataset.collate_fn
    )
    
    # 初始化模型
    model = TestNet().to(device)
    
    # # 由于我们已经提前计算了特征，可以冻结特征提取器的参数
    # for param in model.feature_extractor.parameters():
    #     param.requires_grad = False
    
    # 只优化预测网络的参数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 50
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        # 训练
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        
        # 验证
        val_loss = evaluate(model, test_loader, device, epoch)
        
        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            torch.save(model.prediction_net.state_dict(), "best_prediction_net.pth")
            torch.save(model.feature_extractor.state_dict(), "best_feature_extractor.pth")
            print(f"Saved best model with validation loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()