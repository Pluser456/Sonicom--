import os
import argparse
import torch
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet
from TestNet import TestNet as create_model
from utils import split_dataset, train_one_epoch, evaluate


def setup(rank, world_size):
    """
    初始化分布式训练环境
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    清理分布式训练环境
    """
    dist.destroy_process_group()


def main_worker(rank, world_size, args):
    setup(rank, world_size)
    
    # 设置当前设备
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    if rank == 0 and not os.path.exists("./CNNweights"):
        os.makedirs("./CNNweights")

    # 仅在主进程创建SummaryWriter
    if rank == 0:
        tb_writer = SummaryWriter()
    else:
        tb_writer = None
    
    image_dir = "Ear_image_gray"
    hrtf_dir = "FFT_HRTF"

    dataset_paths = split_dataset(image_dir, hrtf_dir)
    # 获取各个数据集
    train_hrtf_list = dataset_paths['train_hrtf_list']
    test_hrtf_list = dataset_paths['test_hrtf_list']
    left_train = dataset_paths['left_train']
    right_train = dataset_paths['right_train']
    left_test = dataset_paths['left_test']
    right_test = dataset_paths['right_test']

    data_transform = {
        "train": transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        "val": transforms.Compose([  # 验证集可保持原逻辑
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    }

    # 实例化训练数据集
    train_dataset = SonicomDataSet(hrtf_files=train_hrtf_list,
                            left_images=left_train,
                            right_images=right_train,
                            transform=data_transform["train"],
                            mode="left")

    # 实例化验证数据集
    log_mean_hrtf_left = train_dataset.log_mean_hrtf_left
    log_mean_hrtf_right = train_dataset.log_mean_hrtf_right
    val_dataset = SonicomDataSet(hrtf_files=test_hrtf_list,
                            left_images=left_test,
                            right_images=right_test,
                            transform=data_transform["val"],
                            mode="left",
                            calc_mean=False,
                            provided_mean_left=log_mean_hrtf_left,
                            provided_mean_right=log_mean_hrtf_right
                            )

    batch_size = args.batch_size
    
    # 使用DistributedSampler来分配数据给不同GPU
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # 设置工作线程数
    nw = min([os.cpu_count() // world_size, batch_size if batch_size > 1 else 0, 8])
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size*2,
                                              sampler=train_sampler,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size*6,
                                            sampler=val_sampler,
                                            pin_memory=True,
                                            num_workers=nw,
                                            collate_fn=val_dataset.collate_fn)

    # 创建模型并将其包装为DDP模型
    model = create_model().to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    # 仅在主进程打印模型参数信息
    if rank == 0:
        for name, param in model.named_parameters():
            print(f"Layer: {name:30} | Requires Grad: {param.requires_grad}")
    
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)  # 0.05

    # 启用CUDA优化
    torch.backends.cudnn.benchmark = True

    for epoch in range(args.epochs*2):
        # 在每个epoch开始前设置sampler的epoch，确保每个epoch的数据顺序不同
        train_sampler.set_epoch(epoch)
        
        # train
        train_loss = train_one_epoch(model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch,
                                     rank=rank)
        
        # validate
        val_loss = evaluate(model=model,
                            data_loader=val_loader,
                            device=device,
                            epoch=epoch,
                            rank=rank)

        # 仅在主进程记录日志和保存模型
        if rank == 0 and tb_writer is not None:
            tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            # tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], val_loss, epoch)
            # tb_writer.add_scalar(tags[3], val_acc, epoch)
            tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
            
            if epoch % 1 == 0:  # 保存模型
                # 仅保存模块状态而非完整DDP模型
                torch.save(model.module.state_dict(), f"./CNNweights/model-{epoch}.pth")

    cleanup()


def main(args):
    # 获取可用GPU数量
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("检测到只有一个GPU，无法使用多GPU训练")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        # 调用原来的单GPU训练逻辑（代码略）
        # 可以根据需要实现单GPU的备选方案
        return
    
    print(f"使用 {world_size} 个GPU进行分布式训练")
    
    # 使用多进程启动多GPU训练
    mp.spawn(
        main_worker,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)