import os
import argparse

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


# from my_dataset import MyDataSet
from new_dataset import SonicomDataSet, SingleSubjectDataSet
from vit_model import vit_base_patch16_224_in21k as create_model
# from utils import read_split_data, train_one_epoch, evaluate
from utils import split_dataset, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

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

    '''data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}'''
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),  # 直接缩放到 224x224（可能改变长宽比）
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]),
        "val": transforms.Compose([  # 验证集可保持原逻辑
            transforms.Resize(256),
            transforms.CenterCrop(224),
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
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size*6,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)

#num-classes =1000
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        print("weight_dict.state_dict().keys():{}".format(weights_dict.keys()))
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['head.weight', 'head.bias'] # 'pre_logits.fc.weight', 'pre_logits.fc.bias',
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        print("Freezing layers except last two blocks, head, pre_logits, and cross_attn...")
        # 获取最后两层的名称前缀
        last_two_blocks = [f"blocks.{i}" for i in [-2, -1]]  # 自动适配不同深度
        
        for name, para in model.named_parameters():
            # 解冻条件：属于最后两层、head、pre_logits 或 cross_attn
            #在这里解冻!!!解冻名字参考下面代码!!!
            if (
                "norm.weight" in name 
                or "norm.bias" in name 
                or "blocks.11" in name 
                #or "blocks.10" in name 
                or "head" in name 
                or "pre_logits" in name 
                or "pos_proj" in name 
                or "cross_attn" in name  # 新增：解冻交叉注意力层
            ):
                para.requires_grad_(True)
                print(f"Training: {name}")
            else:
                para.requires_grad_(False)
    for name, param in model.named_parameters():#print判断冻结情况
        print(f"Layer: {name:30} | Requires Grad: {param.requires_grad}")
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=0.05)  # 0.05
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf (改成adamw)


    for epoch in range(args.epochs):
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

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.01)


    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符 jx_vit_base_patch16_224_in21k-e5005f0a.pth
    parser.add_argument('--weights', type=str, default='./jx_vit_base_patch16_224_in21k-e5005f0a.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=True)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)