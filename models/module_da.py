import torch
import torch.nn as nn
import torch.nn.functional as F

from .depth_anything_v2.dpt import DepthAnythingV2






def cosine_beta_schedule(timesteps, s=0.008):
    """生成余弦噪声调度"""
    steps = torch.arange(timesteps, dtype=torch.float32) / timesteps
    betas = torch.cos((steps + s) / (1 + s) * torch.pi / 2) ** 2
    betas = betas / betas[0]
    betas = betas * 0.02
    return torch.clamp(betas, 0.0001, 0.9999)

class DoubleConv(nn.Module):
    """双卷积块：Conv -> InstanceNorm -> ReLU -> Conv -> InstanceNorm -> ReLU"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # 替换为 InstanceNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True),  # 替换为 InstanceNorm2d
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetDiffusion(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super(UNetDiffusion, self).__init__()
        self.in_channels = in_channels * 2  # 增加通道以接受 x_lowlight
        self.out_channels = out_channels

        # Encoder
        self.enc1 = DoubleConv(self.in_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.enc3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)

        # Middle
        self.middle = DoubleConv(base_channels * 4, base_channels * 8)

        # Decoder
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_channels * 2, base_channels)

        # Final conv
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Time embedding
        self.time_dim = base_channels * 8
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim)
        )

    def forward(self, x, t, x_lowlight=None, x_normal=None):
        # 时间嵌入
        t = t.unsqueeze(-1).float()  # Shape: (batch_size, 1)
        t_emb = self.time_embed(t)   # Shape: (batch_size, time_dim)

        # 将 x_lowlight 与输入 x 拼接
        if x_lowlight is not None:
            x = torch.cat([x, x_lowlight], dim=1)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        m = self.middle(self.pool(e3))

        # 添加时间嵌入
        t_emb = t_emb.view(-1, self.time_dim, 1, 1)
        m = m + t_emb

        # Decoder
        d3 = self.up3(m)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # 最终输出（预测噪声）
        out = self.final_conv(d1)
        return out, None
    
# class DiffusionModel(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, base_channels=32, timesteps=1000):
#         super(DiffusionModel, self).__init__()
#         self.unet = UNetDiffusion(in_channels, out_channels, base_channels)
#         self.timesteps = timesteps

#         # 余弦噪声调度
#         self.beta = cosine_beta_schedule(timesteps).cuda()
#         self.alpha = 1. - self.beta
#         self.alpha_bar = torch.cumprod(self.alpha, dim=0)

#     def forward(self, x_lowlight, x_normal=None):
#         batch_size = x_lowlight.shape[0]
#         device = x_lowlight.device

#         # 随机时间步
#         t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
#         t_float = t.float() / self.timesteps

#         # 前向加噪
#         x_0 = x_normal if x_normal is not None else x_lowlight
#         alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
#         epsilon = torch.randn_like(x_0)
#         x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

#         # UNet 预测噪声
#         out, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
#         loss = F.mse_loss(out, epsilon) if x_normal is not None else None

#         return out, loss

#     @torch.no_grad()
#     def sample(self, x_lowlight, num_steps=None):
#         """从噪声采样生成增强图像"""
#         if num_steps is None:
#             num_steps = self.timesteps

#         x_t = torch.randn_like(x_lowlight).to(x_lowlight.device)
#         for t in reversed(range(num_steps)):
#             t_float = torch.full((x_lowlight.shape[0],), t / self.timesteps, device=x_lowlight.device)
#             epsilon_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
#             alpha_t = self.alpha[t].view(-1, 1, 1, 1)
#             alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
#             beta_t = self.beta[t].view(-1, 1, 1, 1)
#             x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_t)
#             if t > 0:
#                 x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)

#         return x_t



class DiffusionModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.unet = UNetDiffusion(in_channels, out_channels, base_channels)  # 假设 UNetDiffusion 已定义
        self.timesteps = timesteps

        # 余弦噪声调度，存储到与模型相同的设备
        self.register_buffer('beta', cosine_beta_schedule(timesteps))
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))

    def forward(self, x_lowlight, x_normal=None):
        batch_size = x_lowlight.shape[0]
        device = x_lowlight.device

        # 随机时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        t_float = t.float() / self.timesteps

        # 前向加噪
        x_0 = x_normal if x_normal is not None else x_lowlight
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

        # UNet 预测噪声
        epsilon_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
        loss = F.mse_loss(epsilon_pred, epsilon) if x_normal is not None else None

        return epsilon_pred, loss*100000.0

    @torch.no_grad()
    def sample(self, x_lowlight, num_steps=None, use_ddim=True, eta=0.0):
        """从噪声采样生成去噪后的增强图像，支持 DDIM 和 DDPM"""
        if num_steps is None:
            num_steps = self.timesteps

        device = x_lowlight.device
        x_t = torch.randn_like(x_lowlight).to(device)

        if use_ddim:
            # DDIM 采样
            step_indices = torch.arange(num_steps, dtype=torch.long, device=device) * (self.timesteps // num_steps)
            for i in range(num_steps - 1, -1, -1):
                t = step_indices[i]
                t_float = (t / self.timesteps).float().unsqueeze(0).to(device)
                t_next = step_indices[i - 1] if i > 0 else -1

                # 预测噪声
                epsilon_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)

                # DDIM 去噪公式
                alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
                alpha_bar_t_next = self.alpha_bar[t_next].view(-1, 1, 1, 1) if t_next >= 0 else torch.tensor(1.0, device=device)
                sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
                x_0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_bar_t)
                x_t = torch.sqrt(alpha_bar_t_next) * x_0_pred + torch.sqrt(1 - alpha_bar_t_next - sigma_t ** 2) * epsilon_pred
                if i > 0:
                    x_t += sigma_t * torch.randn_like(x_t)
        else:
            # DDPM 采样
            for t in reversed(range(num_steps)):
                t_float = torch.full((x_lowlight.shape[0],), t / num_steps, device=device)
                epsilon_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
                alpha_t = self.alpha[t * self.timesteps // num_steps].view(-1, 1, 1, 1)
                alpha_bar_t = self.alpha_bar[t * self.timesteps // num_steps].view(-1, 1, 1, 1)
                beta_t = self.beta[t * self.timesteps // num_steps].view(-1, 1, 1, 1)
                x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * epsilon_pred) / torch.sqrt(alpha_t)
                if t > 0:
                    x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)

        return x_t







import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_beta_schedule_(timesteps, s=0.008):
    """余弦噪声调度"""
    steps = torch.arange(timesteps, dtype=torch.float32) / timesteps
    beta = torch.clip(1 - torch.cos(steps * torch.pi / 2) ** 2, 0.0001, 0.9999)
    return beta

class DiffusionModel_(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, timesteps=1000):
        super(DiffusionModel_, self).__init__()
        self.unet = UNetDiffusion(in_channels, out_channels, base_channels)
        self.timesteps = timesteps

        # 余弦噪声调度
        self.beta = cosine_beta_schedule_(timesteps).cuda()
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward(self, x_lowlight, x_normal=None):
        batch_size = x_lowlight.shape[0]
        device = x_lowlight.device

        # 随机时间步
        t = torch.randint(0, self.timesteps, (batch_size,), device=device).long()
        t_float = t.float() / self.timesteps

        # 前向加噪
        x_0 = x_normal if x_normal is not None else x_lowlight
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x_0)
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * epsilon

        # UNet 预测原图 x_0
        x_0_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
        loss = F.mse_loss(x_0_pred, x_0) if x_normal is not None else None

        return x_0_pred, loss * 10000.0 if loss is not None else None

    @torch.no_grad()
    def sample(self, x_lowlight, num_steps=None, use_ddim=True, eta=0.0):
        """从噪声采样生成增强图像，支持 DDIM 和 DDPM"""
        if num_steps is None:
            num_steps = self.timesteps

        device = x_lowlight.device
        x_t = torch.randn_like(x_lowlight).to(device)

        if use_ddim:
            # DDIM 采样
            step_indices = torch.arange(num_steps, dtype=torch.long, device=device) * (self.timesteps // num_steps)
            for i in range(num_steps - 1, -1, -1):
                t = step_indices[i]
                t_float = (t / self.timesteps).float().unsqueeze(0).to(device)
                t_next = step_indices[i - 1] if i > 0 else -1

                # 预测原图 x_0
                x_0_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)

                # DDIM 去噪公式
                alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
                alpha_bar_t_next = self.alpha_bar[t_next].view(-1, 1, 1, 1) if t_next >= 0 else torch.tensor(1.0, device=device)
                sigma_t = eta * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))

                # 根据 x_0_pred 重构 x_t
                x_t = torch.sqrt(alpha_bar_t_next) * x_0_pred + torch.sqrt(1 - alpha_bar_t_next - sigma_t ** 2) * torch.randn_like(x_t)
                if i > 0:
                    x_t += sigma_t * torch.randn_like(x_t)
        else:
            # DDPM 采样
            for t in reversed(range(num_steps)):
                t_float = torch.full((x_lowlight.shape[0],), t / num_steps, device=device)
                # 预测原图 x_0
                x_0_pred, _ = self.unet(x_t, t_float, x_lowlight=x_lowlight)
                alpha_t = self.alpha[t * self.timesteps // num_steps].view(-1, 1, 1, 1)
                alpha_bar_t = self.alpha_bar[t * self.timesteps // num_steps].view(-1, 1, 1, 1)
                beta_t = self.beta[t * self.timesteps // num_steps].view(-1, 1, 1, 1)

                # 根据 x_0_pred 重构 x_t
                epsilon = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_bar_t) * x_0_pred + torch.sqrt(1 - alpha_bar_t) * epsilon
                if t > 0:
                    x_t = (x_t - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * (x_t - torch.sqrt(alpha_bar_t) * x_0_pred)) / torch.sqrt(alpha_t)
                    x_t += torch.sqrt(beta_t) * torch.randn_like(x_t)

        return x_t







import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block2d(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class UNet2D_s(nn.Module):
    def __init__(self, img_channels=3, base_channels=8, time_emb_dim=16):
        super().__init__()
        
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.downs = nn.ModuleList([
            Block2d(base_channels, base_channels, time_emb_dim),
            Block2d(base_channels, base_channels * 2, time_emb_dim),
            Block2d(base_channels * 2, base_channels * 4, time_emb_dim),
            # Block2d(base_channels * 4, base_channels * 8, time_emb_dim)
        ])
        
        # Pooling layers
        self.pool = nn.MaxPool2d(2)
        
        # Bridge
        self.bridge = Block2d(base_channels * 4, base_channels * 8, time_emb_dim)
        # self.bridge = Block(base_channels * 2, base_channels * 4, time_emb_dim)

        
        # Upsampling path
        self.ups = nn.ModuleList([
            # Block2d(base_channels * 8 + base_channels * 8, base_channels * 8, time_emb_dim),
            Block2d(base_channels * 8 + base_channels * 4, base_channels * 4, time_emb_dim),
            Block2d(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim),
            Block2d(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        ])
        
        # Upsampling layers
        self.ups_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolution
        self.final_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, time):
        # Time embedding
        time_emb = self.time_embedding(time)
        
        # Initial convolution
        x = self.init_conv(x)
        
        # Skip connections
        skip_connections = []
        
        # Downsampling
        for down in self.downs:
            x = down(x, time_emb)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bridge
        x = self.bridge(x, time_emb)
        
        # Upsampling
        for idx, up in enumerate(self.ups):
            x = self.ups_sample(x)
            skip = skip_connections[-(idx+1)]
            
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply upsampling block
            x = up(x, time_emb)
        
        # Final convolution
        return self.final_conv(x)






class Diffusion_2d(nn.Module):
    def __init__(self, base_channels=8, num_diffusion_steps=5):
        super(Diffusion_2d, self).__init__()
        # 原始3D卷积网络层

        # 扩散模型参数
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_scheduler = torch.linspace(0.1, 0.0001, num_diffusion_steps)  # 噪声调度器

        self.unet = UNet2D_s()


# 正向加噪函数
    def add_noise(self, data, noise_level):
        noise = torch.randn_like(data) * noise_level  # 按当前步噪声水平加噪
        noisy_data = data + noise
        return noisy_data  # 返回加噪后的数据和噪声

# 损失函数定义
    def diffusion_loss(self,predicted_noise, true_noise):
        return F.mse_loss(predicted_noise, true_noise)
    

    # def forward(self, conv0):  #训练sd
    #     # 初始化输入([1, 64, 1, 48, 96])
    #     # max_value = conv0.max().item()
    #     # min_value = conv0.min().item()

    #     # conv0 = conv0
    #     total_loss = 0  
    #     step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重

    #     conv0_ = conv0.clone()#.squeeze(2)
    #     # print(conv0.max())
    #     for step in range(self.num_diffusion_steps):
    #         # 添加噪声

    # # 加噪处理
    #         noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度

    #         noisy_input = self.add_noise(conv0_, noise_level)

    #         # noisy_input = conv0
    #         t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()

    #         conv0_ = self.unet(noisy_input,t)

    #     return conv0_, total_loss

        # return conv0, total_loss
    

    def forward_(self, conv0,gt):  #训练sd
        # 初始化输入([1, 64, 1, 48, 96])
        # max_value = conv0.max().item()
        # min_value = conv0.min().item()

        # conv0 = conv0
        total_loss = 0  
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重

        conv0_ = conv0.clone()#.squeeze(2)

        # print(conv0.max())
        for step in range(self.num_diffusion_steps):
            # 添加噪声

    # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
            # print(noise_level)

            noisy_input = self.add_noise(conv0, noise_level)

            # noisy_input = conv0
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()

            predicted_conv_loss = self.unet(noisy_input,t)

            # if step ==1:
            #     return predicted_conv_loss, total_loss

            weight = step_weights[4 - step]  # 根据噪声调度器获取当前步权重
            loss = self.diffusion_loss(predicted_conv_loss, gt)
            weighted_loss = weight * loss
            total_loss += weighted_loss

            # conv0_ = predicted_conv_loss


            # conv0_ = self.unet(noisy_input,t)

        return conv0_, total_loss


    def forward(self, conv0,gt):   #二阶段sd
        # 初始化输入
        # max_value = conv0.max().item()
        # min_value = conv0.min().item()
        #conv0就是输入图片

        total_loss = 0  
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重

        # conv0_ = conv0.clone()
        for step in range(self.num_diffusion_steps):
            # 添加噪声

    # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度

            noisy_input = self.add_noise(conv0, noise_level)

            # noisy_input = conv0
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()

            predicted_conv_loss = self.unet(noisy_input,t)

            if step ==1:
                return predicted_conv_loss, total_loss

            # weight = step_weights[2 - step]  # 根据噪声调度器获取当前步权重
            # loss = self.diffusion_loss(predicted_conv_loss, conv0)
            # weighted_loss = weight * loss
            # total_loss += weighted_loss
            conv0 = predicted_conv_loss

        return conv0, total_loss
    




























class ConvGRUCell2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(ConvGRUCell2, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size, padding=1)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size, padding=1)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = torch.cat((x, h), dim=1)
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        C = f.shape[1]
        r, u = torch.split(f, C // 2, 1)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = F.sigmoid(rn)
        uns = F.sigmoid(un)
        return rns, uns

    def output(self, x, h, r, u):
        f = torch.cat((x, r * h), dim=1)
        o = self.output_conv(f)
        on = self.output_norm(o)
        return on

    def forward(self, x, h = None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)
        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        output = u * h + (1 - u) * y
        return output, output


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)

        self.conv_gates = nn.Sequential(nn.Conv2d(self.input_channels + self.hidden_channels, 2 * self.hidden_channels,
                                    kernel_size=self.kernel_size, stride=1,
                                    padding=self.padding, bias=True))
                                    # nn.GroupNorm(1, 2 * self.hidden_channels, 1e-5, True))

        self.convc = nn.Sequential(nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels,
                               kernel_size=self.kernel_size, stride=1,
                               padding=self.padding, bias=True))
                               # nn.GroupNorm(1, self.hidden_channels, 1e-5, True))

    def forward(self, x, h):
        N, C, H, W = x.shape[0],x.shape[1],x.shape[2], x.shape[3]
        HC = self.hidden_channels
        if(h is None):
            h = torch.zeros((N, HC, H, W), dtype=torch.float, device=x.device)

        input = torch.cat((x, h), dim=1)
        gates = self.conv_gates(input)

        reset_gate, update_gate = torch.chunk(gates, dim=1, chunks=2)

        # activation
        reset_gate = torch.sigmoid(reset_gate)
        update_gate = torch.sigmoid(update_gate)

        # print(reset_gate)
        # concatenation
        input = torch.cat((x, reset_gate * h), dim=1)

        # convolution
        conv = self.convc(input)

        # activation
        conv = torch.tanh(conv)

        # soft update
        output = update_gate * h + (1 - update_gate) * conv

        return output, output


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 * self.hidden_channels, self.kernel_size, 1,
                              self.padding)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, h, c):
        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, A.size()[1] // self.num_features, dim=1)
        i = torch.sigmoid(ai)    #input gate
        f = torch.sigmoid(af)    #forget gate
        o = torch.sigmoid(ao)    #output
        g = torch.tanh(ag)       #update_Cell

        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c, o


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)

    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvTransBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransBnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvTransReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)


    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class ConvGn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvGn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return self.gn(self.conv(x))


class ConvTransGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, output_pad=1):
        super(ConvTransGnReLU, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad, output_padding=output_pad, bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return F.relu(self.gn(self.conv(x)), inplace=True)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class GlobalPoolingModule(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(GlobalPoolingModule, self).__init__()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv3 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=1, bias=False)

    def forward(self, x):
        block1 = self.conv1(x)
        block2 = self.conv2(block1)
        block3 = self.Sigmoid(block2)
        block4 = block1 + block3 * block2
        block5 = self.conv3(block4)

        return block5


class ChannelAttentionModule(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d()
        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=1, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1, x2):
        x = x1 + x2
        block1 = self.avg_pool(x)
        block2 = F.relu(self.conv1(block1),  inplace=True)
        block3 = self.sigmoid(self.conv2(block2))
        block4 = x + block3 * x


        return block4


def homo_warping_float(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # Tcw
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1) # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    if torch.__version__ <= "1.3.0":
        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                       padding_mode='zeros')
    else:
        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) # Tcw
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1).double()  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth,
                                                                                            -1).double()  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy.float()

    if torch.__version__ <= "1.3.0":
        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                       padding_mode='zeros')
    else:
        warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                       padding_mode='zeros', align_corners=True)
        
    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    else:
        depth_values = F.interpolate(depth_values, [p.shape[2], p.shape[3]], mode='bilinear', align_corners=False)


    depth = torch.sum(p * depth_values, 1)
    return depth


def get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth=192.0, min_depth=0.0):
    #shape, (B, H, W)
    #cur_depth: (B, H, W)
    #return depth_range_values: (B, D, H, W)
    cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel)  # (B, H, W)
    cur_depth_max = (cur_depth + ndepth / 2 * depth_inteval_pixel)
    # cur_depth_min = (cur_depth - ndepth / 2 * depth_inteval_pixel).clamp(min=0.0)   #(B, H, W)
    # cur_depth_max = (cur_depth_min + (ndepth - 1) * depth_inteval_pixel).clamp(max=max_depth)

    assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
    new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, H, W)

    depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=cur_depth.device,
                                                                  dtype=cur_depth.dtype,
                                                                  requires_grad=False).reshape(1, -1, 1,
                                                                                               1) * new_interval.unsqueeze(1))

    return depth_range_samples


def get_depth_range_samples(cur_depth, exp_var,ndepth, depth_inteval_pixel, device, dtype, shape,
                           max_depth=192.0, min_depth=0.0):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or (B, D)
    #return depth_range_samples: (B, D, H, W)
    eps = 1e-12
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) + (torch.arange(0, ndepth, device=device, dtype=dtype,
                                                                       requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)

    else:
        # depth_range_samples = get_cur_depth_range_samples(cur_depth, ndepth, depth_inteval_pixel, shape, max_depth, min_depth)

        low_bound = cur_depth - exp_var
        high_bound = cur_depth + exp_var

        # assert exp_var.min() >= 0, exp_var.min()
        assert ndepth > 1

        step = (high_bound - low_bound) / (float(ndepth) - 1)
        new_samps = []
        for i in range(int(ndepth)):
            new_samps.append(low_bound + step * i + eps)

        depth_range_samples = torch.cat(new_samps, 1)

    return depth_range_samples

from .att import AttentionBlock

class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )


        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)





        DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        encoder = 'vitb'  # or 'vits', 'vitb', 'vitg'

        self.model = DepthAnythingV2(**model_configs[encoder])
        # self.model.load_state_dict(torch.load(f'.depth_anything_v2_{encoder}.pth', map_location='cuda'))
        # self.model.load_state_dict(torch.load('/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/models/depth_anything_v2_vitb.pth', map_location='cuda'))



        # self.model.load_state_dict(
        #     torch.load('/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/models/depth_anything_v2_vitb.pth',
        #                map_location='cuda'),
        #     strict=False
        # )
        # pretrained_dict = torch.load('/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/models/depth_anything_v2_vitb.pth',
        #                map_location='cuda')

        pretrained_dict = torch.load('/opt/data/private/cas11.10/models/depth_anything_v2_vitb.pth',
                       map_location='cuda')
        # for k, v in pretrained_dict.items():
        #     # if pattern.match(k):
        #     #     print(f"Matched parameter: {k}")
        #     print(k)

        model_dict = self.model.state_dict()
        # 将预训练权重中与当前模型结构匹配的权重加载


        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           # k in model_dict and k.startswith("pretrained") and v.size() == model_dict[k].size()}
        k in model_dict and k.startswith("pretrained")}



        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        self.model = self.model.to(DEVICE).eval()
        for param in self.model.parameters():
            param.requires_grad = False



        # self.diffusion = Diffusion_2d()

        # pretrained_diffusion = torch.load('/opt/data/private/cas11.10/st_bl_sd_nos_0.1_los_sd_feature_/model_000029.ckpt',
        #                map_location='cuda')["model"]

        # # for k, v in pretrained_diffusion.items():
        # #     print(k)
        # # for name, param in pretrained_diffusion.items():
        # #     # print("....................")
        # #     print(name)
        # #     # print(param)

        # model_diffusion = self.diffusion.state_dict()

        # pretrained_diffusion = {k.replace('module.feature.diffusion.', ''): v for k, v in pretrained_diffusion.items() if k.startswith("module.feature.diffusion")}
        # print(".........")
        # # print(pretrained_diffusion)
        # for name, param in self.diffusion.named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        #     # param.requires_grad = False
        # model_diffusion.update(pretrained_diffusion)
        # self.diffusion.load_state_dict(model_diffusion, strict=True)


    def forward(self, nview_idx, x, test):

    # def forward(self, nview_idx, x):
        outputs = {}
        deps = {}

        # if not test and nview_idx == 0:
        #     x, _ = self.diffusion(x)

        # with torch.no_grad():
        #     x, loss = self.diffusion (x)
        # print(x.shape)
        # raw_img = cv2.imread('c.jpg')
        # print(depth.shape)
        # x_min = torch.min(x)
        # x_max = torch.max(x)
        # x_norm = (x - x_min) / (x_max - x_min)
        # if :
        #     x,_ = self.diffusion(x)
            # print(x)
        with torch.no_grad():
            depth3, depth2, depth1 = self.model.infer_image(x)  # HxW raw depth map in numpy
        deps["stage1"] = depth3
        deps["stage2"] = depth2
        deps["stage3"] = depth1
        # deps= self.model.infer_image(x)  # HxW raw depth map in numpy

        # return deps


        # depth = self.pipe(x_norm, **self.pipe_kwargs)
        # depth_pred: np.ndarray = depth.prediction

        # source_outputs = None
        # deps_out = []
        # # print(nview_idx)
        # if nview_idx==0:
        #     # source_outputs = depth_pred.float()
        #
        #     c1 = self.c0(depth_pred.float())
        #     c2 = self.c1(c1)
        #     c3 = self.c2(c2)
        #     out2 = self.o1(c1)  # torch.Size([1, 1, 512, 640])
        #     out1 = self.o2(c2)  # torch.Size([1, 1, 256, 320])
        #     out0 = self.o3(c3)  # torch.Size([1, 1, 128, 160])
        #
        #     deps_out.append(out0)
        #     # print(out0.shape)
        #     deps_out.append(out1)
        #     # print(out1.shape)
        #
        #     deps_out.append(out2)
        #     # print(out2.shape)
        #     # print(len(out))
        #



            #
            #
            # import matplotlib.pyplot as plt
            #
            # # 假设你已经有了两个变量 depth_pred 和 depth_colored
            # # depth_pred 是一个 NumPy 数组，depth_colored 是一个 PIL Image 对象
            #
            # # 展示 depth_pred
            # plt.figure(figsize=(12, 6))
            # plt.subplot(1, 3, 1)  # 这将创建一个1行2列的子图，当前是第一个
            # plt.imshow(x.squeeze(0).cpu().permute(1, 2, 0))  # 使用灰度色彩映射
            # plt.colorbar()  # 显示色彩条
            # plt.title('Depth Prediction')
            #
            # # 展示 depth_colored
            # plt.subplot(1, 3, 2)  # 这是第二个子图
            # plt.imshow(depth1.squeeze(0).cpu().permute(1, 2, 0)  , cmap='viridis')
            # plt.title('Colored Depth Image')
            #
            # plt.subplot(1, 3, 3)  # 这是第二个子图
            # plt.imshow(depth_pred.squeeze(0).cpu().permute(1, 2, 0), cmap='viridis')
            # plt.title('Colored Depth Image')
            # plt.show()  # 显示图像



        # print("x")
        # print(x.shape)
        # print("depth")
        # print(depth_pred.shape)



        # cs1 = self.c0(depth)
        # cs2 = self.c1(cs1)
        # cs3 = self.c2(cs2)
        # outs2 = self.o1(cs1)  # torch.Size([1, 1, 512, 640])
        # outs1 = self.o2(cs2)  # torch.Size([1, 1, 256, 320])
        # outs0 = self.o3(cs3)  # torch.Size([1, 1, 128, 160])
        #
        # deps_out.append(outs0)
        # # print(out0.shape)
        # deps_out.append(outs1)
        # # print(out1.shape)
        #
        # deps_out.append(outs2)
        #
        # deps["stage1"] = outs0
        # deps["stage2"] = outs1
        # deps["stage3"] = outs2




        conv0 = self.conv0(x)
        # conv0 = self.convX(torch.cat((x,depth), dim=1))
        # (1, 8, 384,768)


        # conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)  # (1, 16, 192,384)
        conv2 = self.conv2(conv1) # (1, 32, 96,192)

        intra_feat = conv2
        # outputs = {}
        out = self.out1(intra_feat) # (1, 32, 96,192)
        outputs["stage1"] = out


        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out


                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)  # (1, 32, 192,384)
                out = self.out2(intra_feat)# (1, 16, 192,384)
                outputs["stage2"] =out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0) # (1, 32, 384,768)
                out = self.out3(intra_feat) # (1, 8, 384,768)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out
        # print(len(out))
        return outputs, deps


class CostRMVSNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRMVSNet, self).__init__()
        self.conv_gru1 = ConvGRUCell(in_channels, base_channels, 3)
        self.conv_gru2 = ConvGRUCell(base_channels, 4, 3)
        self.conv_gru3 = ConvGRUCell(4, 2, 3)
        self.conv2d = nn.Conv2d(2, 1, 3, 1, 1)

    def forward(self, x, state1, state2, state3):
        reg_cost1, state1 = self.conv_gru1(-x, state1)
        reg_cost2, state2 = self.conv_gru2(reg_cost1, state2)
        reg_cost3, state3 = self.conv_gru3(reg_cost2, state3)
        reg_cost = self.conv2d(reg_cost3)
        return reg_cost, state1, state2, state3


class CostREDNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostREDNet, self).__init__()
        self.conv_gru1 = ConvGRUCell(in_channels, base_channels, 3)
        self.conv_gru2 = ConvGRUCell(base_channels, base_channels*2, 3)
        self.conv_gru3 = ConvGRUCell(base_channels*2, base_channels*4, 3)
        self.conv_gru4 = ConvGRUCell(base_channels*4, base_channels*8, 3)
        self.conv1 = convReLU(in_channels, base_channels*2, 3, 2, 1)
        self.conv2 = convReLU(base_channels*2, base_channels*4, 3, 2, 1)
        self.conv3 = convReLU(base_channels*4, base_channels*8, 3, 2, 1)
        self.upconv3 = ConvTransReLU(base_channels*8, base_channels*4, 3, 2, 1, 1)
        self.upconv2 = ConvTransReLU(base_channels*4, base_channels*2, 3, 2, 1, 1)
        self.upconv1 = ConvTransReLU(base_channels*2, base_channels, 3, 2, 1, 1)
        self.upconv2d = nn.ConvTranspose2d(base_channels, 1, 3, 1, 1, 1)

    def forward(self, x, state1, state2, state3, state4):
        # Recurrent Regularization
        conv_cost1 = self.conv1(-x)
        conv_cost2 = self.conv2(conv_cost1)
        conv_cost3 = self.conv3(conv_cost2)
        reg_cost4, state4 = self.conv_gru4(conv_cost3, state4)
        up_cost3 = self.upconv3(reg_cost4)
        reg_cost3, state3 = self.conv_gru3(conv_cost2, state3)
        up_cost33 = torch.add(up_cost3, reg_cost3)
        up_cost2 = self.upconv2(up_cost33)
        reg_cost2, state2 = self.conv_gru2(conv_cost1, state2)
        up_cost22 = torch.add(up_cost2, reg_cost2)
        up_cost1 = self.upconv1(up_cost22)
        reg_cost1, state1 = self.conv_gru1(-volume_variance, state1)
        up_cost11 = torch.add(up_cost1, reg_cost1)
        reg_cost = self.upconv2d(up_cost11)
        return reg_cost, state1, state2, state3, state4


class PropagationNet(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(PropagationNet, self).__init__()
        self.base_channels = base_channels
        #self.img_conv = ImageConv(base_channels)

        self.conv1 = nn.Sequential(
            Conv2d(in_channels, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 2, 3, 1, padding=1),
            nn.Conv2d(base_channels * 2, 9, 3, padding=1, bias=False)
        )

        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=1, padding=0)

    def forward(self, depth, img_features):
        #img_featues = self.img_conv(img)
        img_conv = img_features

        x = self.conv1(img_conv)
        prob = F.softmax(x, dim=1)

        depth_pad = F.pad(depth, (1, 1, 1, 1), mode='replicate')
        depth_unfold = self.unfold(depth_pad)

        b, c, h, w = prob.size()
        prob = prob.view(b, 9, h * w)

        result_depth = torch.sum(depth_unfold * prob, dim=1)
        result_depth = result_depth.view(b, 1, h, w)
        return result_depth


if __name__ == "__main__":
    # some testing code, just IGNORE it
    from datasets import find_dataset_def
    from torch.utils.data import DataLoader
    import numpy as np
    import cv2

    MVSDataset = find_dataset_def("dtu_yao")
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 256)
    dataloader = DataLoader(dataset, batch_size=2)
    item = next(iter(dataloader))

    imgs = item["imgs"][:, :, :, ::4, ::4].cuda()
    proj_matrices = item["proj_matrices"].cuda()
    mask = item["mask"].cuda()
    depth = item["depth"].cuda()
    depth_values = item["depth_values"].cuda()

    imgs = torch.unbind(imgs, 1)
    proj_matrices = torch.unbind(proj_matrices, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

    warped_imgs = homo_warping(src_imgs[0], src_projs[0], ref_proj, depth_values)

    cv2.imwrite('../tmp/ref.png', ref_img.permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)
    cv2.imwrite('../tmp/src.png', src_imgs[0].permute([0, 2, 3, 1])[0].detach().cpu().numpy()[:, :, ::-1] * 255)

    for i in range(warped_imgs.shape[2]):
        warped_img = warped_imgs[:, :, i, :, :].permute([0, 2, 3, 1]).contiguous()
        img_np = warped_img[0].detach().cpu().numpy()
        cv2.imwrite('../tmp/tmp{}.png'.format(i), img_np[:, :, ::-1] * 255)


    # generate gt
    def tocpu(x):
        return x.detach().cpu().numpy().copy()


    ref_img = tocpu(ref_img)[0].transpose([1, 2, 0])
    src_imgs = [tocpu(x)[0].transpose([1, 2, 0]) for x in src_imgs]
    ref_proj_mat = tocpu(ref_proj)[0]
    src_proj_mats = [tocpu(x)[0] for x in src_projs]
    mask = tocpu(mask)[0]
    depth = tocpu(depth)[0]
    depth_values = tocpu(depth_values)[0]

    for i, D in enumerate(depth_values):
        height = ref_img.shape[0]
        width = ref_img.shape[1]
        xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
        #print("yy", yy.max(), yy.min())
        yy = yy.reshape([-1])
        xx = xx.reshape([-1])
        X = np.vstack((xx, yy, np.ones_like(xx)))
        # D = depth.reshape([-1])
        # print("X", "D", X.shape, D.shape)

        X = np.vstack((X * D, np.ones_like(xx)))
        X = np.matmul(np.linalg.inv(ref_proj_mat), X)
        X = np.matmul(src_proj_mats[0], X)
        X /= X[2]
        X = X[:2]

        yy = X[0].reshape([height, width]).astype(np.float32)
        xx = X[1].reshape([height, width]).astype(np.float32)

        warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
        # warped[mask[:, :] < 0.5] = 0

        cv2.imwrite('../tmp/tmp{}_gt.png'.format(i), warped[:, :, ::-1] * 255)
