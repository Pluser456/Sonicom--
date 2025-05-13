import torch
import numpy as np
import pytorch_lightning as pl
from collections import OrderedDict
from argparse import ArgumentParser
from torch import nn


def _conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1): #创建卷积块，stride为步长，padding为填充，2倍下采样，ep：[1, in_channels, 32, 32] 转换为 [1, out_channels, 16, 16]
    block = OrderedDict([
        ('conv2d', nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),
        ('bn2d', nn.BatchNorm2d(out_channels)), #归一化
        ('act', nn.LeakyReLU()) #激活函数
    ])
    return nn.Sequential(block)

def _conv_block_transp(in_channels, out_channels, kernel_size=3, stride=1, padding=1, is_last=False): #创建反卷积块，2倍上采样，ep：[1, in_channels, 16, 16] 转换为 [1, out_channels, 32, 32]
    block = [
        ('upsamp2d', nn.UpsamplingNearest2d(scale_factor=2)), #2倍上采样，复制像素操作，ep：[1, in_channels, 16, 16] -> [1, in_channels, 32, 32]
        ('convtr2d', nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)),#步长为1，图像维度不变，通道数改变，ep：[1, in_channels, 32, 32] -> [1, out_channels, 32, 32]
    ]
    if is_last:
        block.append(('act', nn.Sigmoid())) #最后一层使用sigmoid激活函数
        #block.append(('act', nn.LeakyReLU()))
    else:
        block.extend([
            ('bn2d', nn.BatchNorm2d(out_channels)),
            ('act', nn.LeakyReLU())
        ])
    return nn.Sequential(OrderedDict(block))

def _lin_block(in_size, out_size, dropout_rate=0.2): #创建线性块
    block = OrderedDict([
        ('lin', nn.Linear(in_size, out_size)),
        ('act', nn.LeakyReLU()),
        #('drop', nn.Dropout(dropout_rate))
    ])
    return nn.Sequential(block)

def _calc_output_shape(input_shape, model): #计算输出形状
    in_tensor = torch.zeros(1, *input_shape)
    with torch.no_grad():
        out_tensot = model(in_tensor)
    return list(out_tensot.shape)[1:]


class ConvVAE(nn.Module):
    def __init__(self, input_shape, encoder_channels, latent_size, decoder_channels):
        super().__init__()
        #print(input_shape) [1,793,108]
        assert len(input_shape) == 3 #检查输入形状是否为三维
        assert type(encoder_channels) == list 
        assert type(latent_size) == int
        assert type(decoder_channels) == list
        self.latent_size = latent_size #隐变量z的维度
        self.enc = Encoder(input_shape, encoder_channels, latent_size)
        self.dec = Decoder(self.enc.conv_out_shape, decoder_channels, latent_size,output_shape=input_shape)

    def forward(self, x):
        means, log_var = self.enc(x)
        z = self.reparameterize(means, log_var)
        recon_x = self.dec(z)
        return recon_x, means, log_var, z #返回重构的图像，均值，方差，隐变量

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var) #计算标准差
        eps = torch.randn_like(std) #生成噪声
        return mu + eps * std


class Encoder(nn.Module):
    def __init__(self, input_shape, channels, latent_size):
        super().__init__()
        channels = [input_shape[0]] + channels #将输入通道数添加到channels列表的开头
        self.conv_stack = nn.Sequential() 
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.conv_stack.add_module(name=f'block_{i}', module=_conv_block(in_channels, out_channels)) #添加卷积块
        self.conv_out_shape = _calc_output_shape(input_shape, self.conv_stack)
        linear_size = np.prod(self.conv_out_shape) #conv最后一层的大小，连到线性层得到维度为latent_size的z的均值和方差
        self.linear_means = _lin_block(linear_size, latent_size)
        self.linear_log_var = _lin_block(linear_size, latent_size)

    def forward(self, x):
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1) #展平为一维，便于全连接,过程如下：
        #[batch_size, channels, height, width] -> dim=1:[batch_size, channels * height * width]
        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)
        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, conv_out_shape, channels, latent_size,output_shape):
        super().__init__()
        #channels = channels + channels[-1:]
        linear_size = np.prod(conv_out_shape)
        self.conv_out_shape = conv_out_shape 
        #print(conv_out_shape) #[128,50,7]
        self.output_shape = output_shape
        self.linear_stack = _lin_block(latent_size, linear_size)#反线性层
        self.conv_stack = nn.Sequential()
        for i, (in_channels, out_channels) in enumerate(zip(channels[:-1], channels[1:])):
            self.conv_stack.add_module(name=f'block_{i}', module=_conv_block_transp(in_channels, out_channels))#添加反卷积块
        self.conv_stack.add_module(name='block_out', module=_conv_block_transp(channels[-1], 1, is_last=True))#添加最后一层的反卷积块
        
        conv_output_size = _calc_output_shape(conv_out_shape, self.conv_stack)
        #print(conv_output_size) #[1, 800, 112]
        self.crop_height = conv_output_size[1] - output_shape[1]  # 800 - 793 = 7
        self.crop_width = conv_output_size[2] - output_shape[2] #4
        self.conv_stack.add_module(name='crop', module=nn.Conv2d(1, 1, kernel_size=(self.crop_height + 1, self.crop_width + 1), stride=(1,1), padding=0))
        #print(_calc_output_shape(conv_out_shape, self.conv_stack)) #[1, 793, 108]

    def forward(self, z):
        x = self.linear_stack(z)
        x = x.view(x.shape[0], *self.conv_out_shape)#展平
        x = self.conv_stack(x)
        return x


# convolutional variational autoencoder
class VAECfg(pl.LightningModule):
    model_name = 'VAE_conv'

    def __init__(self, input_size, cfg):
        super().__init__()
        self.save_hyperparameters() #保存超参数
        self.grad_freq = 1 #50个epoch后记录梯度
        self.fig_freq = 1 #10个epoch后记录图
        self.kl_coeff = cfg['kl_coeff'] #cfg文件中kl散度的系数  
        input_shape = [cfg['input_channels']] + input_size #input_shape为
        encoder_channels = cfg['encoder_channels']
        latent_size = cfg['latent_size']
        decoder_channels = cfg['decoder_channels']
        self.vae = ConvVAE(input_shape, encoder_channels, latent_size, decoder_channels)

    @staticmethod
    def add_model_specific_args(parent_parser):#添加模型特定的参数
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--model_cfg_path', type=str, default='./configs/models/vae_conv/small.json')
        return parser

    def loss_function(self, ear_true, ear_pred, means, log_var, z):
        mse = torch.nn.functional.mse_loss(ear_pred, ear_true, reduction='sum') / ear_true.size(0)
        kld = -0.5 * torch.sum(1 + log_var - means.pow(2) - log_var.exp()) * self.kl_coeff / ear_true.size(0)
        loss = mse + kld #计算损失，MSE损失衡量的是重构误差，KL散度衡量的是潜在变量分布与标准正态分布之间的差异
        return mse, kld, loss

    def forward(self, x):
        return self.vae(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)#学习率为1e-5
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5624, patience=50, cooldown=25),
            'monitor': 'val_loss'
        }
        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):#记录训练集的损失
        _, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        self.log('train_recon_loss', mse) #记录MSE损失
        self.log('train_kl', kld) 
        self.log('train_loss', loss) 
        return loss

    def validation_step(self, batch, batch_idx): #记录验证集的损失
        _, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        self.log('val_recon_loss', mse)
        self.log('val_kl', kld)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        results, losses = self._shared_eval(batch, batch_idx)
        mse, kld, loss = losses
        ear_pred, means, log_var, z = results
        ear_true, labels = batch
        # log metrics
        # TODO add metrics: SD
        logs = {
            'test_recon_loss': mse,
            'test_kl': kld,
            'test_loss': loss
        }
        self.log_dict(logs) #记录测试集的损失
        # log reconstructions
        ear_true, ear_pred = ear_true.cpu(), ear_pred.cpu() #移到CPU上
        img = self.get_pred_ear_figure(ear_true, ear_pred, n_cols=8)
        self.logger.experiment.add_image(f'test/ears_{batch_idx:04}', img, self.current_epoch)#记录图像，batch，epoch

    def training_epoch_end(self, outputs):
        # log gradients
        if self.current_epoch % self.grad_freq == 0: #每50个epoch记录梯度
            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(name, params, self.current_epoch)#记录参数名称和参数的值，epoch
        # log figures
        if self.current_epoch % self.fig_freq == 0: #每10个epoch记录图
            # run prediction
            ear_true = self.example_input_array.to(self.device) #获取样本输入
            self.eval() #设置为评估模式
            with torch.no_grad(): #禁止梯度计算
                ear_pred, means, log_var, z = self.forward(ear_true)
            self.train() #设置为训练模式
            ear_true = ear_true.to(self.device) #移到GPU上，方便后续计算
            # generate figure
            img = self.get_pred_ear_figure(ear_true, ear_pred)
            self.logger.experiment.add_image('Valid/ears', img, self.current_epoch)

    def _shared_eval(self, batch, batch_idx):#计算损失和结果
        ear_true = batch['hrtf']
        results = self.forward(ear_true)#计算结果
        losses = self.loss_function(ear_true, *results)#计算损失
        return results, losses

    def get_pred_ear_figure(self, ear_true, ear_pred, n_cols=8): #n_cols: 每行显示的图像列数
        bs = ear_true.shape[0] # 获取输入图像批次的大小（即batch_size）
        n_rows = max(bs, 1) // n_cols # 计算需要多少行图像
        imgs = []
        for i in range(n_rows):
            sl = slice(i * n_cols, min((i + 1) * n_cols, bs))# 计算当前行图像的索引范围
            img_true = torch.dstack(ear_true[sl].unbind()) 
            #unbind()函数将tensor拆分成多个张量,dstack()函数将多个张量堆叠成一个张量,ep:[8,1,32,32]->8x[1,32,32]->[1,32,32,8]
            img_pred = torch.dstack(ear_pred[sl].unbind()) 
            img = torch.hstack((img_true, img_pred))#ep:2x[1,32,32,8]->[1,32,32,16]
            imgs.append(img)
        img = torch.hstack(imgs)
        return img #返回原始图像和重构图像的对比
