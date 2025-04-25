import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from new_dataset import SonicomDataSet
from TestNet import TestNet as create_model
from utils import split_dataset, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./CNNweights") is False:
        os.makedirs("./CNNweights")

    tb_writer = SummaryWriter()
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
            transforms.RandomHorizontalFlip(),
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
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4]) 
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size*3,
                                               shuffle=True,
                                               pin_memory=True,
                                            #    num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size*6,
                                             shuffle=False,
                                             pin_memory=True,
                                            #  num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model().to(device)


    for name, param in model.named_parameters():#print判断冻结情况
        print(f"Layer: {name:30} | Requires Grad: {param.requires_grad}")
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.01)  # 0.05
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf (改成adamw)


    for epoch in range(args.epochs*5):
        # train
        train_loss = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)


        # validate
        val_loss = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if epoch % 5 == 0:
            # 保存模型
            torch.save(model.state_dict(), "./CNNweights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)


    parser.add_argument('--model-name', default='', help='create model name')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)