import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_da import *
import numpy as np
import matplotlib.pyplot as plt

Align_Corners_Range = False


def cas_mvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=True)

    for stage_key, stage_inputs in inputs.items():
        if "stage" not in stage_key:
            continue
        # print(stage_key)


        if stage_key == "stage3":
            # print(inputs["refined_depth"].shape)

            depth_est =inputs["refined_depth"]
            depth_est1 =inputs["refined_depth"]
        else:
            # print(stage_inputs["depth"].shape)

            depth_est = stage_inputs["depth"]


        # depth_est = stage_inputs["depth"]
        # depth_est1 = inputs["refined_depth"]
        # depth_est = stage_inputs["refined_depth"]

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        # min_val = depth_gt[mask].min().item()
        # depth_max = depth_gt[mask].max().item()
        # print(f"depth_gt 范围: min={min_val}, max={max_val}")


        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        # prob_loss = Adaptive_Multi_Modal_Cross_Entropy_Loss(depth_est, depth_gt, mask, int(depth_max)) #+ depth_loss
        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss =total_loss + depth_loss_weights[stage_idx] * depth_loss
            # total_loss += depth_loss_weights[stage_idx] * diffusion_loss*0.1`
        else:
            total_loss += 1.0 * depth_loss



        if stage_key == "stage3":
            depth_loss1 = F.smooth_l1_loss(depth_est1[mask], depth_gt[mask], reduction='mean')

        # print(inputs["refined_depth"].shape)


    # total_loss =total_loss +  inputs["lossdf"] 
    return total_loss, depth_loss1
    # return total_loss, depth_loss1

# from __future__ import print_function
# from torch.autograd import Variable
# import math
# import numpy as np





# groud-truth Laplace distribution
def LaplaceDisp2Prob(Gt, maxdisp=192, m=1, n=9):
    N, H, W = Gt.shape
    b = 0.8

    Gt = torch.unsqueeze(Gt, 1)
    disp = torch.arange(maxdisp, device=Gt.device)
    disp = disp.reshape(1, maxdisp, 1, 1).repeat(N, 1, H, W)
    cost = -torch.abs(disp - Gt) / b

    return F.softmax(cost, dim=1)


def Adaptive_Multi_Modal_Cross_Entropy_Loss(x, disp, mask, maxdisp, m=1, n=9, top_k=9, epsilon=3, min_samples=1):
    # maxdisp = int(maxdisp)
    # print(maxdisp)
    # disp[disp >= 192] = 0

    N, H, W = disp.shape
    patch_h, patch_w = m, n
    disp_unfold = F.unfold(F.pad(disp, (patch_w // 2, patch_w // 2, patch_h // 2, patch_h // 2), mode='reflect'),
                           (patch_h, patch_w)).view(N, patch_h * patch_w, H, W)
    disp_unfold_clone = torch.clone(disp_unfold)

    mask_cluster = torch.zeros((N, patch_h * patch_w, patch_h * patch_w, H, W), device=disp.device).bool()
    for index in range(patch_h * patch_w):
        if index == 0:
            d_min = d_max = disp.unsqueeze(1)
        else:
            disp_unfold = disp_unfold * ~mask_cluster[:, index - 1, ...]
            d_min = d_max = torch.max(disp_unfold, dim=1, keepdim=True)[0]
        mask_cluster[:, index, ...] = (disp_unfold > (d_min - epsilon).clamp(min=1e-6)) & (
                    disp_unfold < (d_max + epsilon).clamp(max=maxdisp - 1))
        while True:
            d_min = torch.min(disp_unfold * mask_cluster[:, index, ...] + ~mask_cluster[:, index, ...] * 192, dim=1,
                              keepdim=True)[0]
            d_max = torch.max(disp_unfold * mask_cluster[:, index, ...], dim=1, keepdim=True)[0]
            mask_new = (disp_unfold > (d_min - epsilon).clamp(min=0)) & (
                        disp_unfold < (d_max + epsilon).clamp(max=maxdisp - 1))
            if mask_new.sum() == mask_cluster[:, index, ...].sum():
                break
            else:
                mask_cluster[:, index, ...] = mask_new

    disp_cluster = torch.mean(disp_unfold_clone.unsqueeze(1).repeat(1, patch_h * patch_w, 1, 1, 1) * mask_cluster,
                              dim=2) * (patch_h * patch_w) / torch.sum(mask_cluster, dim=2).clamp(min=1)

    # GT = torch.zeros((N, patch_h * patch_w, maxdisp, H, W), device=disp.device)
    GT = torch.zeros((N, patch_h * patch_w, maxdisp, H, W), device=disp.device)

    for index in range(patch_h * patch_w):
        if index == 0:
            GT[:, index, ...] = LaplaceDisp2Prob(disp, maxdisp)
        else:
            GT[:, index, ...] = LaplaceDisp2Prob(disp_cluster[:, index, ...], maxdisp)

    mask_cluster = torch.sum(mask_cluster, dim=2, keepdim=True)
    mask_cluster[mask_cluster < min_samples] = 0

    w_cluster = 0.2 / (mask_cluster.sum(dim=1, keepdim=True) - 1).clamp(min=1) * mask_cluster
    w_cluster[:, 0, ...] += 0.8 - 0.2 / (mask_cluster.sum(dim=1, keepdim=False) - 1).clamp(min=1)

    top_k_values, top_k_indices = torch.topk(w_cluster, k=top_k, dim=1)
    w_cluster.fill_(0)
    w_cluster.scatter_(dim=1, index=top_k_indices, src=top_k_values)
    w_cluster = w_cluster / w_cluster.sum(dim=1, keepdim=True).clamp(min=1)

    GT = (GT * w_cluster).sum(dim=1, keepdim=False)
    GT = GT.detach_()
    num = mask.sum()
    x = torch.log(x + 1e-30)
    mask = torch.unsqueeze(mask, 1).repeat(1, maxdisp, 1, 1)
    # print(x.shape)
    # print(GT.shape)
    # print(mask.shape)
    # print(x[mask].shape)  # 检查通过 mask 后的 GT 形状
    # torch.Size([1, 326, 96, 192])
    # torch.Size([1, 326, 96, 192])
    # torch.Size([5651536])

    loss = - (GT[mask] * x[mask]).sum() / num

    return loss












class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 池化到 [B, C, 1, 1, 1]
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # [B, C]
        y = self.fc(y).view(b, c, 1, 1, 1)  # [B, C, 1, 1, 1]
        return x * y.expand_as(x)

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)  # 假设最后3维是空间维度
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, D, H, W]
        y = torch.cat([avg_out, max_out], dim=1)  # [B, 2, D, H, W]
        y = self.conv(y)  # [B, 1, D, H, W]
        return x * self.sigmoid(y)

# 自适应融合模块（用InstanceNorm3d替代BatchNorm3d）
class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptiveFusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels // 2),  # 替换为InstanceNorm3d
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        return self.fusion(x)
    


# 完整的融合类
class FeatureFusion3d(nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels):
        super(FeatureFusion3d, self).__init__()
        self.ca1 = ChannelAttention(in_channels1)
        self.ca2 = ChannelAttention(in_channels2)
        self.sa = SpatialAttention()
        self.fusion_module = AdaptiveFusion(in_channels1 + in_channels2, out_channels)

    def forward(self, feature1, feature2):
        # 通道注意力
        feature1_ca = self.ca1(feature1)
        feature2_ca = self.ca2(feature2)

        # 空间注意力
        feature1_sa = self.sa(feature1_ca)
        feature2_sa = self.sa(feature2_ca)

        # 拼接
        fused = torch.cat((feature1_sa, feature2_sa), dim=1)

        # 自适应融合
        fused_feature = self.fusion_module(fused)
        return fused_feature











class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.lamb = 1.5
        # self.sd = True

        # self.fusion = FeatureFusion()
        # self.fusion1 = FeatureFusion(total_channels=96, reduction=16)  # 第一组: 32 + 64
        # self.fusion2 = FeatureFusion(total_channels=80, reduction=16)  # 第二组: 16 + 64
        # self.fusion3 = FeatureFusion(total_channels=72, reduction=16)

        # self.fusion1 = FeatureFusion3d(in_channels1=32, in_channels2=64, out_channels=96)  # 第一组: 32 + 64
        # self.fusion2 = FeatureFusion3d(in_channels1=16, in_channels2=64, out_channels=80)
        # self.fusion3 = FeatureFusion3d(in_channels1=8, in_channels2=64, out_channels=72)   # 第三组: 8 + 64
    def forward(self, features,deps_feature_stage, proj_matrices, stage_idx,depth_values, num_depth, cost_regularization, prob_volume_init=None):
        # print(proj_matrices.shape)



        proj_matrices = torch.unbind(proj_matrices, 1)



        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # print(depth_values.shape)
        # print(features[1].shape)
        # print(deps_feature_stage[1].shape)


        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        ref_deps, src_deps = deps_feature_stage[0], deps_feature_stage[1:]


        ref_deps = ref_deps.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        # print(ref_deps.shape)



        deps_sum = ref_deps
        deps_sq_sum = ref_deps ** 2
        del ref_deps
        for src_de, src_proj in zip(src_deps, src_projs):
            #warpped features
            warped_deps = homo_warping_float(src_de, src_proj, ref_proj, depth_values)

            if self.training:
                deps_sum = deps_sum + warped_deps
                deps_sq_sum = deps_sq_sum + warped_deps ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                deps_sum += warped_deps
                deps_sq_sum += warped_deps.pow_(2)  # the memory of warped_volume has been modified
            del warped_deps
        deps_sum = deps_sum.div_(num_views)
        deps_variance = deps_sq_sum.div_(num_views).sub_(deps_sum.pow_(2))





        # # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        # print(ref_volume.shape)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            warped_volume = homo_warping_float(src_fea, src_proj, ref_proj, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_sum = volume_sum.div_(num_views)
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.pow_(2))



        # print(volume_variance.shape,deps_variance.shape)
        # print(out_volume.shape)
        # volume_variance = torch.cat((volume_variance, out_volume), dim=1)
        
        volume_variance = torch.cat((volume_variance, deps_variance), dim=1)

        # volume_variance = self.fusion(volume_variance, deps_variance)


        # cost_reg = volume_variance
        cost_reg,diffusion_loss = cost_regularization(volume_variance)



        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init

        prob_volume = F.softmax(prob_volume_pre, dim=1)
#96,192

        depth = depth_regression(prob_volume, depth_values=depth_values)

      
        samp_variance = (depth_values - depth.unsqueeze(1)) ** 2


        exp_variance = self.lamb * torch.sum(samp_variance * prob_volume, dim=1, keepdim=False) ** 0.5

        # return {"depth": depth,  "photometric_confidence": photometric_confidence, 'variance': exp_variance, 'prob_volume': prob_volume, 'depth_values':depth_values}
        return {"depth": depth, 'variance': exp_variance, 'prob_volume': prob_volume, 'depth_values':depth_values,"diffusion_loss":diffusion_loss}

class CostRegNet_old(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(CostRegNet_old, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x



class CostRegNet_noise(nn.Module):
    def __init__(self, in_channels, base_channels=8, num_diffusion_steps=10):
        super(CostRegNet_noise, self).__init__()

        # 原始3D卷积网络层
        self.conv0 = ConvBnReLU3D(in_channels, 8)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2)

        # 引入Attention

        self.conv2 = ConvBnReLU3D(16, 16)
        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)
        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        # 改为使用插值上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv7 = nn.Sequential(
            self.upsample,
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True))

        self.conv9 = nn.Sequential(
            self.upsample,
            nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True))

        self.conv11 = nn.Sequential(
            self.upsample,
            nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.LeakyReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        # 扩散过程的步数
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_scale = torch.linspace(0.5, 0, num_diffusion_steps)

    def forward(self, x):
        # print(x.shape)
        conv0 = self.conv0(x)

        for step in range(self.num_diffusion_steps):
            noise = torch.randn_like(conv0) * self.noise_scale[step]
            conv0 = conv0 + noise  # 添加噪声

        # 使用Attention
            conv1 = self.conv1(conv0)
            # attn_output, _ = self.attention(conv1, conv1, conv1)

            conv2 = self.conv2(conv1)
            conv4 = self.conv4(self.conv3(conv2))

            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)
        x = self.prob(x)
        # print(x.shape)

        return x




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


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch)
        )
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act1 = nn.SiLU()
        self.act2 = nn.SiLU()
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, t):
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet3D(nn.Module):
    def __init__(self, img_channels=8, base_channels=8, time_emb_dim=16):
        super().__init__()
        # Time embedding
        self.time_embedding = SinusoidalPositionEmbeddings(time_emb_dim)
        # Initial convolution
        self.init_conv = nn.Conv3d(img_channels, base_channels, 3, padding=1)
        # Downsampling path
        self.downs = nn.ModuleList([
            Block(base_channels, base_channels, time_emb_dim),
            Block(base_channels, base_channels * 2, time_emb_dim),
            Block(base_channels * 2, base_channels * 4, time_emb_dim),
            # Block(base_channels * 4, base_channels * 8, time_emb_dim)
        ])
        # Pooling layers
        self.pool = nn.MaxPool3d(2)
        # Bridge
        self.bridge = Block(base_channels * 4, base_channels * 8, time_emb_dim)
        # Upsampling path
        self.ups = nn.ModuleList([
            # Block(base_channels * 16 + base_channels * 8, base_channels * 8, time_emb_dim),
            Block(base_channels * 8 + base_channels * 4, base_channels * 4, time_emb_dim),
            Block(base_channels * 4 + base_channels * 2, base_channels * 2, time_emb_dim),
            Block(base_channels * 2 + base_channels, base_channels, time_emb_dim)
        ])
        # Upsampling layers
        self.ups_sample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        # Final convolution
        self.final_conv = nn.Conv3d(base_channels, img_channels, 1)

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
            skip = skip_connections[-(idx + 1)]
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            # Apply upsampling block
            x = up(x, time_emb)
        # Final convolution
        return self.final_conv(x)

import random
class Diffusion_new(nn.Module):
    def __init__(self, base_channels=8, num_diffusion_steps=1):
        super(Diffusion_new, self).__init__()
        # 原始3D卷积网络层
        # 扩散模型参数
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_scheduler = torch.linspace(0.1, 0.01, num_diffusion_steps)  # 噪声调度器


        self.unet = UNet3D()

        # self.conv = nn.Conv3d(16, 8, kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm3d(8)
        # self.relu = nn.ReLU()

    # 正向加噪函数
    def add_noise(self, data, noise_level):
        noise = torch.randn_like(data) * noise_level  # 按当前步噪声水平加噪
        noisy_data = data + noise
        return noisy_data, data  # 返回加噪后的数据和噪声

    # 损失函数定义
    # def diffusion_loss(self, predicted_noise, true_noise):
    #     return F.mse_loss(predicted_noise, true_noise)
    def diffusion_loss(self, predicted_noise, true_noise):
        return F.mse_loss(predicted_noise, true_noise) + \
            0.3 * F.l1_loss(predicted_noise, true_noise)  # 引入 L1 损失



    def forward_(self, conv0): #训练sd
        # 初始化输入
        max_value = conv0.max().item()
        min_value = conv0.min().item()
        total_loss = 0
        conv0_old = conv0.clone()
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重
        for step in range(self.num_diffusion_steps):
            # 添加噪声
            # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度

            noisy_input, no_noise = self.add_noise(conv0, noise_level)
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            # 卷积降噪过程
            predicted_conv_loss = self.unet(noisy_input, t)
            predicted_conv_loss = torch.clamp(predicted_conv_loss, max=max_value, min=min_value)
            # 绘制原始和预测图像
            # predicted_conv = noisy_input - predicted_noise


            # weight = step_weights[step]  # 根据噪声调度器获取当前步权重
            # loss = self.diffusion_loss(predicted_conv_loss, conv0)
            # weighted_loss = weight * loss
            # total_loss += weighted_loss

            conv0 = predicted_conv_loss


        # 随机选择返回 predicted_conv_loss 或者 conv0


        if random.choice([True, False]):
            return predicted_conv_loss, total_loss
        else:
            return conv0_old, total_loss


        # return conv0, total_loss

    def forward(self, conv0): #测试
        # 初始化输入
        max_value = conv0.max().item()
        min_value = conv0.min().item()
        total_loss = 0
        conv0_old = conv0.clone()
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重
        for step in range(self.num_diffusion_steps):
            # 添加噪声
            # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
        
            noisy_input, no_noise = self.add_noise(conv0, noise_level)
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            # 卷积降噪过程
            predicted_conv_loss = self.unet(noisy_input, t)
            predicted_conv_loss = torch.clamp(predicted_conv_loss, max=max_value, min=min_value)
            # 绘制原始和预测图像
            # predicted_conv = noisy_input - predicted_noise
        
        
            # weight = step_weights[step]  # 根据噪声调度器获取当前步权重
            # loss = self.diffusion_loss(predicted_conv_loss, conv0)
            # weighted_loss = weight * loss
            # total_loss += weighted_loss
        
            conv0 = predicted_conv_loss


        # 随机选择返回 predicted_conv_loss 或者 conv0


        # if random.choice([True, False]):
        #     return predicted_conv_loss, total_loss
        # else:
        #     return conv0_old, total_loss


        return conv0, total_loss



    def forward_(self, conv0): #sd预测
        # 初始化输入
        max_value = conv0.max().item()
        min_value = conv0.min().item()
        total_loss = 0
        conv0_old = conv0.clone()
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重
        for step in range(self.num_diffusion_steps):
            # 添加噪声
            # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
            noisy_input, no_noise = self.add_noise(conv0, noise_level)
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            # 卷积降噪过程
            predicted_conv_loss = self.unet(noisy_input, t)
            predicted_conv_loss = torch.clamp(predicted_conv_loss, max=max_value, min=min_value)
            # 绘制原始和预测图像
            # predicted_conv = noisy_input - predicted_noise
            weight = step_weights[2-step]  # 根据噪声调度器获取当前步权重
            loss = self.diffusion_loss(predicted_conv_loss, conv0_old)
            weighted_loss = weight * loss
            total_loss += weighted_loss

            conv0 = predicted_conv_loss

        total_loss += self.diffusion_loss(conv0, conv0_old)

        return conv0, total_loss




    def forward_(self, conv0):
        # 初始化输入
        max_value = conv0.max().item()
        min_value = conv0.min().item()
        total_loss = 0
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重
        for step in range(self.num_diffusion_steps):
            # 添加噪声
            # 加噪处理
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
            if step != 0:
                predicted_conv2, _ = self.add_noise(predicted_conv2, noise_level)
            noisy_input, no_noise = self.add_noise(conv0, noise_level)
            # noisy_input = conv0
            t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            # 卷积降噪过程
            predicted_conv_loss = self.unet(noisy_input, t)

            predicted_conv_loss = torch.clamp(predicted_conv_loss, max=max_value, min=min_value)


            # 绘制原始和预测图像
            # predicted_conv = noisy_input - predicted_noise
            weight = step_weights[step]  # 根据噪声调度器获取当前步权重

            loss = self.diffusion_loss(predicted_conv_loss, no_noise)
            weighted_loss = weight * loss
            total_loss += weighted_loss


            if step == 0:
                predicted_conv2 = predicted_conv_loss
            else:
                # with torch.no_grad():
                predicted_conv2 = self.unet(predicted_conv2, t)
            # 监督损失
            # loss = F.mse_loss(denoised, conv0)  # 与无噪声特征 conv0 比较
            # conv0 =torch.clamp(predicted_conv, min=min_value, max=max_value)   # 更新去噪特征
            # conv0 = predicted_conv2
        #     del noisy_input, predicted_conv
        #     torch.cuda.empty_cache()
        #     with torch.no_grad():
        #         for step in range(self.num_diffusion_steps):
        #             # 添加噪声
        #     # 加噪处理
        #             noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
        #             noisy_input, _ = self.add_noise(conv0, noise_level)
        #             # noisy_input = conv0
        #             t = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
        #             # 卷积降噪过程
        #             conv0 = self.unet(noisy_input, t)
        # # 绘制原始和预测图像
        #             # predicted_conv = noisy_input - predicted_noise
        return torch.clamp(predicted_conv2, max=max_value,min=min_value), total_loss




import math
class Diffusion(nn.Module):
    def __init__(self, base_channels=8, num_diffusion_steps=10):
        super(Diffusion, self).__init__()
        # 原始3D卷积网络层
        self.conv1 = ConvBnReLU3D(base_channels, base_channels * 2, stride=2)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2)
        self.conv3 = ConvBnReLU3D(base_channels * 2, base_channels * 4, stride=2)
        self.conv4 = ConvBnReLU3D(base_channels * 4, base_channels * 4)
        self.conv5 = ConvBnReLU3D(base_channels * 4, base_channels * 8, stride=2)
        self.conv6 = ConvBnReLU3D(base_channels * 8, base_channels * 8)

        # 上采样层
        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.Conv3d(16, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.Conv3d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.LeakyReLU(inplace=True)
        )

        # 扩散模型参数
        self.num_diffusion_steps = num_diffusion_steps
        # self.noise_scheduler = torch.linspace(1, 0.01, num_diffusion_steps)  # 噪声调度器


        t = torch.arange(0, num_diffusion_steps, dtype=torch.float32)
        self.noise_scheduler = torch.cos(t / num_diffusion_steps * math.pi / 2) ** 2

        self.time_embed = nn.Sequential(
            nn.Linear(1, 16),  # 保持中间层为 16
            nn.ReLU(),
            nn.Linear(16, base_channels)  # 输出调整为 base_channels（8）
        )

        self.time_embed = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),  # 卷积捕获时间步模式
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, base_channels, kernel_size=3, padding=1)  # 输出维度与 base_channels 匹配
        )

    # forward 中处理时间步

    # 正向加噪函数
    def add_noise(self, data, noise_level):
        noise = torch.randn_like(data) * noise_level  # 按当前步噪声水平加噪
        noisy_data = data + noise
        return noisy_data, noise  # 返回加噪后的数据和噪声

    # 损失函数定义
    # def diffusion_loss(self, predicted_noise, true_noise):
    #     return F.mse_loss(predicted_noise, true_noise)

    def diffusion_loss(self, predicted_noise, true_noise):
        return F.mse_loss(predicted_noise, true_noise) + \
            0.3 * F.l1_loss(predicted_noise, true_noise)  # 引入 L1 损失

    def forward(self, conv0):
        # 初始化输入
        total_loss = 0
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重
        conv0clone= conv0.clone()
        for step in range(self.num_diffusion_steps):
            # 添加噪声
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
            # print(noise_level)
            noisy_input, noise = self.add_noise(conv0clone, noise_level)

            # 时间步嵌入
            # time_step = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            # time_embedding = self.time_embed(time_step.unsqueeze(0))
            # noisy_input = noisy_input + time_embedding.view(1, -1, 1, 1, 1)

            time_step = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float().unsqueeze(
                0).unsqueeze(-1)
            time_embedding = self.time_embed(time_step).view(1, -1, 1, 1, 1)
            # print(time_embedding.shape)

            noisy_input = noisy_input + time_embedding

            # 反向传播降噪过程
            conv1 = self.conv1(noisy_input)
            conv2 = self.conv2(conv1)
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            denoised = noisy_input + self.conv11(x)

            # 计算预测噪声
            predicted_noise = noisy_input - denoised
            weight = step_weights[step]  # 根据噪声调度器获取当前步权重
            loss = self.diffusion_loss(predicted_noise, noise)
            weighted_loss = weight * loss
            total_loss += weighted_loss

            # conv0clone = denoised

        # conv0 = denoised
        # 输出结果
        return conv0, total_loss





    def forward_(self, conv0):
        # 初始化输入
        total_loss = 0
        step_weights = self.noise_scheduler / self.noise_scheduler.sum()  # 归一化权重

        for step in range(self.num_diffusion_steps):
            # 添加噪声
            noise_level = self.noise_scheduler[step]  # 获取当前时间步噪声强度
            noisy_input, noise = self.add_noise(conv0, noise_level)

            # 时间步嵌入
            time_step = torch.tensor([step / self.num_diffusion_steps], device=conv0.device).float()
            time_embedding = self.time_embed(time_step.unsqueeze(0))
            conv0 = conv0 + time_embedding.view(1, -1, 1, 1, 1)

            # 卷积降噪过程
            conv1 = self.conv1(noisy_input)
            conv2 = self.conv2(conv1)
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            denoised = conv0 + self.conv11(x)

            # 计算预测噪声
            predicted_noise = noisy_input - denoised
            weight = step_weights[step]  # 根据噪声调度器获取当前步权重
            loss = self.diffusion_loss(predicted_noise, noise)
            weighted_loss = weight * loss
            total_loss += weighted_loss

            # 更新去噪特征
            conv0 = denoised

        # 输出结果
        return conv0, total_loss


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels=8, num_diffusion_steps=10):
        super(CostRegNet, self).__init__()
        # 原始3D卷积网络层
        self.conv0 = ConvBnReLU3D(in_channels, base_channels)
        self.conv1 = ConvBnReLU3D(base_channels, base_channels * 2, stride=2)
        self.conv2 = ConvBnReLU3D(base_channels * 2, base_channels * 2)
        self.conv3 = ConvBnReLU3D(base_channels * 2, base_channels * 4, stride=2)
        self.conv4 = ConvBnReLU3D(base_channels * 4, base_channels * 4)
        self.conv5 = ConvBnReLU3D(base_channels * 4, base_channels * 8, stride=2)
        self.conv6 = ConvBnReLU3D(base_channels * 8, base_channels * 8)

        # 上采样层
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv7 = nn.Sequential(
            self.upsample,
            nn.Conv3d(base_channels * 8, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, base_channels * 4),
            # nn.BatchNorm3d(32),

            nn.LeakyReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            self.upsample,
            nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, base_channels * 2),
            # nn.BatchNorm3d(16),
            nn.LeakyReLU(inplace=True)
        )
        self.conv11 = nn.Sequential(
            self.upsample,
            nn.Conv3d(base_channels * 2, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, base_channels),
            # nn.BatchNorm3d(8),

            nn.LeakyReLU(inplace=True)
        )
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1)

        # 扩散模型参数


        # self.diffusion = Diffusion_new()

    def forward(self, x):
        # 初始化输入
        conv0 = self.conv0(x)

        # with torch.no_grad():
        #     conv0_diffusion, diffusion_loss = self.diffusion(conv0)

        # 正向传播
        # conv1 = self.conv1(conv0_diffusion)

        conv1 = self.conv1(conv0)

        conv2 = self.conv2(conv1)
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        diffusion_loss = 0
        return x, diffusion_loss


class CostRegNet_(nn.Module):
    def __init__(self, in_channels, base_channels=8, num_diffusion_steps=4):
        super(CostRegNet_, self).__init__()

        # 原始3D卷积网络层
        self.conv0 = ConvBnReLU3D(in_channels, 8)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2)

        self.conv2 = ConvBnReLU3D(16, 16)
        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)
        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        # 改为使用插值上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.conv7 = nn.Sequential(
            self.upsample,
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.LeakyReLU(inplace=True))

        self.conv9 = nn.Sequential(
            self.upsample,
            nn.Conv3d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.LeakyReLU(inplace=True))

        self.conv11 = nn.Sequential(
            self.upsample,
            nn.Conv3d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(2, 8),
            nn.LeakyReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)


    def forward(self, x):
        conv0 = self.conv0(x)
        # conv0 = self.att(conv0)

        conv1 = self.conv1(conv0)

        # conv1 = self.att2(conv1)

        conv2 = self.conv2(conv1)

        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x




class CostRegNet_1(nn.Module):
    def __init__(self, in_channels, base_channels=8, num_diffusion_steps=4):
        super(CostRegNet, self).__init__()
        # 原始3D卷积网络层
        self.conv0 = ConvBnReLU3D(in_channels, 8)
        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)
        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)
        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        # 扩散过程的步数
        self.num_diffusion_steps = num_diffusion_steps
        self.noise_scale = torch.linspace(1.0, 0.1, num_diffusion_steps)  # 定义噪声尺度，逐步减小

    def forward(self, x):

        conv0 = self.conv0(x)

        for step in range(self.num_diffusion_steps):
            # 在扩散过程中注入噪声

            noise = torch.randn_like(conv0) * self.noise_scale[step]


            conv0 = conv0 + noise  # 添加噪声

            # 3D卷积网络的正则化处理

            conv2 = self.conv2(self.conv1(conv0))
            conv4 = self.conv4(self.conv3(conv2))
            x = self.conv6(self.conv5(conv4))
            x = conv4 + self.conv7(x)
            x = conv2 + self.conv9(x)
            x = conv0 + self.conv11(x)

        x = self.prob(x)

        return x


import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


class CannyEdgeDetector(nn.Module):
    def __init__(self, low_threshold=0.1, high_threshold=0.2, kernel_size=5, sigma=1.0):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        # 创建高斯核
        kernel_x = self._create_gaussian_kernel(kernel_size, sigma)
        kernel_y = kernel_x.transpose(0, 1)

        # 注册为缓冲区，这样它们会被保存在模型中
        self.register_buffer('gaussian_kernel_x', kernel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('gaussian_kernel_y', kernel_y.unsqueeze(0).unsqueeze(0))

        # Sobel 算子
        self.register_buffer('sobel_kernel_x', torch.FloatTensor([[-1, 0, 1],
                                                                  [-2, 0, 2],
                                                                  [-1, 0, 1]]).unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_kernel_y', torch.FloatTensor([[-1, -2, -1],
                                                                  [0, 0, 0],
                                                                  [1, 2, 1]]).unsqueeze(0).unsqueeze(0))

    @staticmethod
    def _create_gaussian_kernel(kernel_size, sigma):
        """创建高斯核"""
        x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
        x = x.view(kernel_size, 1)
        y = x.t()
        gaussian = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.sum()

    def _rgb_to_grayscale(self, image):
        """将 RGB 图像转换为灰度图"""
        weights = torch.tensor([0.299, 0.587, 0.114], device=image.device)
        return (image * weights.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)

    def forward(self, x):
        """
        前向传播函数
        Args:
            x: 输入图像张量，形状为 (B, 3, H, W)
        Returns:
            edge_map: 边缘图，形状为 (B, 1, H, W)
        """
        with torch.no_grad():
            # 转换为灰度图
            x = self._rgb_to_grayscale(x)

            # 高斯模糊
            padding = self.gaussian_kernel_x.shape[-1] // 2
            x = F.conv2d(x, self.gaussian_kernel_x, padding=padding)
            x = F.conv2d(x, self.gaussian_kernel_y, padding=padding)

            # 计算梯度
            grad_x = F.conv2d(x, self.sobel_kernel_x, padding=1)
            grad_y = F.conv2d(x, self.sobel_kernel_y, padding=1)

            # 计算梯度幅值和方向
            magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            direction = torch.atan2(grad_y, grad_x)

            # 非极大值抑制
            angle = torch.round((direction * 180. / torch.pi) / 45.) * 45

            nms = torch.zeros_like(magnitude)
            for batch in range(magnitude.shape[0]):
                for i in range(1, magnitude.shape[2] - 1):
                    for j in range(1, magnitude.shape[3] - 1):
                        angle_val = angle[batch, 0, i, j].item()
                        mag = magnitude[batch, 0, i, j].item()

                        # 根据梯度方向选择邻居像素
                        if angle_val in [-180, 0, 180]:
                            neighbors = [magnitude[batch, 0, i, j - 1], magnitude[batch, 0, i, j + 1]]
                        elif angle_val in [-135, 45]:
                            neighbors = [magnitude[batch, 0, i - 1, j + 1], magnitude[batch, 0, i + 1, j - 1]]
                        elif angle_val in [-90, 90]:
                            neighbors = [magnitude[batch, 0, i - 1, j], magnitude[batch, 0, i + 1, j]]
                        else:  # -45 or 135
                            neighbors = [magnitude[batch, 0, i - 1, j - 1], magnitude[batch, 0, i + 1, j + 1]]

                        if mag >= max(neighbors):
                            nms[batch, 0, i, j] = mag

            # 双阈值处理
            high_mask = nms > self.high_threshold
            low_mask = nms > self.low_threshold

            # 滞后阈值处理
            edge_map = torch.zeros_like(nms)
            edge_map[high_mask] = 1

            # 连接边缘
            for batch in range(edge_map.shape[0]):
                changed = True
                while changed:
                    changed = False
                    for i in range(1, edge_map.shape[2] - 1):
                        for j in range(1, edge_map.shape[3] - 1):
                            if low_mask[batch, 0, i, j] and not edge_map[batch, 0, i, j]:
                                if torch.any(edge_map[batch, 0, i - 1:i + 2, j - 1:j + 2] == 1):
                                    edge_map[batch, 0, i, j] = 1
                                    changed = True

            return 1 - edge_map  # 反转边缘（边缘为黑色）


def extract_edges(image: torch.Tensor, crop_size_h: int, crop_size_w: int) -> torch.Tensor:
    """
    使用纯 PyTorch 实现的 Canny 边缘检测器提取边缘。

    Args:
        image: 输入图像，形状为 (1, 3, H, W)，需要是 PyTorch 张量
        crop_size_h: 裁剪后的图像高度
        crop_size_w: 裁剪后的图像宽度

    Returns:
        边缘张量，形状为 (1, 1, H, W)
    """
    # detector = CannyEdgeDetector().to(image.device)
    detector = CannyEdgeDetector().to(image.device)

    edges = detector(image)

    # 如果需要调整大小
    if (edges.shape[2], edges.shape[3]) != (crop_size_h, crop_size_w):
        edges = F.interpolate(edges, size=(crop_size_h, crop_size_w), mode='bilinear', align_corners=False)

    return edges


# 边缘提取函数 (Canny)
def extract_edges_old(image, crop_size_h, crop_size_w):
    """
    使用 Canny 边缘检测器提取边缘。
    Args:
        image: 输入图像，形状为 (1, 3, H, W)，需要是 PyTorch 张量。
        crop_size_h: 裁剪后的图像高度。
        crop_size_w: 裁剪后的图像宽度。
    Returns:
        edges: 边缘张量，形状为 (1, 1, H, W)。
    """
    device = image.device  # 获取图像所在的设备（CPU 或 CUDA）

    # 去除 batch 维度，形状从 (1, 3, H, W) -> (3, H, W)
    image = image.squeeze(0)

    # 转为 numpy 格式并调整为 (H, W, 3)，适配 OpenCV
    image = image.permute(1, 2, 0).cpu().numpy()  # 转到 CPU 和 NumPy 格式 (H, W, 3)

    # 将均值归一化的数据恢复到标准范围 [0, 255]
    image = (image - image.min()) / (image.max() - image.min()) * 255.0

    # 转为灰度图
    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Canny 边缘检测
    edges = cv2.Canny(gray.astype(np.uint8), 100, 220)

    # 归一化边缘到 [0, 1]，并反转边缘
    edges = edges.astype(np.float32) / 255.0
    edges = 1.0 - edges  # 反转边缘（边缘为黑色）

    # 调整形状为 (1, 1, H, W) 以适配后续张量操作
    edges = edges.reshape((1, 1, crop_size_h, crop_size_w))

    # 返回 CUDA 张量，转移到与输入图像相同的设备
    return torch.tensor(edges, dtype=torch.float32, device=device)

def visualize_propagation_with_histograms(input_data, output_data, title_prefix=""):
    """
    可视化 input_data、output_data 和它们的变化，并输出每个图的直方图。
    Args:
        input_data: 初始输入张量。
        output_data: 经过 propagate 优化后的张量。
        title_prefix: 标题前缀。
    """
    input_data_np = input_data.squeeze().cpu().detach().numpy()
    output_data_np = output_data.squeeze().cpu().detach().numpy()
    difference = output_data_np - input_data_np

    plt.figure(figsize=(18, 10))

    # 原始输入
    plt.subplot(3, 3, 1)
    plt.imshow(input_data_np, cmap='viridis')
    plt.colorbar()
    plt.title(f"{title_prefix} Input Data")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.hist(input_data_np.ravel(), bins=30, color='blue', alpha=0.7)
    plt.title("Input Data Histogram")

    # 优化后的输出
    plt.subplot(3, 3, 2)
    plt.imshow(output_data_np, cmap='viridis')
    plt.colorbar()
    plt.title(f"{title_prefix} Output Data")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.hist(output_data_np.ravel(), bins=30, color='green', alpha=0.7)
    plt.title("Output Data Histogram")

    # 变化值
    plt.subplot(3, 3, 3)
    plt.imshow(difference, cmap='coolwarm')  # 使用冷暖色调突出变化
    plt.colorbar()
    plt.title(f"{title_prefix} Change (Output - Input)")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.hist(difference.ravel(), bins=30, color='red', alpha=0.7)
    plt.title("Change Histogram")

    plt.tight_layout()
    plt.show()


# 边缘感知传播
def propagate1(input_data, dlr, drl, dud, ddu, nlr, nrl, nud, ndu, dim):
    """
    使用边缘权重在四个方向传播优化。
    Args:
        input_data: 输入张量，形状为 (B, 1, H, W)。
        dlr, drl, dud, ddu: 各方向的传播权重，形状为 (B, 1, H, W)。
        dim: 通道数（这里固定为 1）。
    Returns:
        output_data: 优化后的张量，形状为 (B, 1, H, W)。
    """
    B, _, H, W = input_data.shape

    # 左到右传播
    x = torch.zeros((B, 1, H, 1), device=input_data.device)  # 初始化左侧填充
    current_data = torch.cat([x, input_data], dim=3)  # (B, 1, H, W+1)
    current_data = current_data[:, :, :, :W]  # 裁剪到原始宽度
    output_data = current_data * dlr + input_data * (1 - dlr)

    # 右到左传播
    x = torch.zeros((B, 1, H, 1), device=input_data.device)  # 初始化右侧填充
    current_data = torch.cat([output_data, x], dim=3)  # (B, 1, H, W+1)
    current_data = current_data[:, :, :, 1:]  # 裁剪到原始宽度
    output_data = current_data * drl + output_data * (1 - drl)

    # 上到下传播
    x = torch.zeros((B, 1, 1, W), device=input_data.device)  # 初始化顶部填充
    current_data = torch.cat([x, output_data], dim=2)  # (B, 1, H+1, W)
    current_data = current_data[:, :, :H, :]  # 裁剪到原始高度
    output_data = current_data * dud + output_data * (1 - dud)

    # 下到上传播
    x = torch.zeros((B, 1, 1, W), device=input_data.device)  # 初始化底部填充
    current_data = torch.cat([output_data, x], dim=2)  # (B, 1, H+1, W)
    current_data = current_data[:, :, 1:, :]  # 裁剪到原始高度
    output_data = current_data * ddu + output_data * (1 - ddu)
    return output_data


def propagate(input_data, dlr, drl, dud, ddu, nlr, nrl, nud, ndu, dim):
    """
    使用边缘权重和归一化权重在四个方向传播优化。
    Args:
        input_data: 输入张量，形状为 (B, 1, H, W)。
        dlr, drl, dud, ddu: 各方向的传播权重，形状为 (B, 1, H, W)，对应 W_lr, W_rl, W_ud, W_du。
        nlr, nrl, nud, ndu: 各方向的归一化权重，形状为 (B, 1, H, W)，对应 N_lr, N_rl, N_ud, N_du。
        dim: 通道数（这里固定为 1）。
    Returns:
        output_data: 优化后的张量，形状为 (B, 1, H, W)。
    """
    B, _, H, W = input_data.shape

    # 左到右传播
    x = torch.zeros((B, 1, H, 1), device=input_data.device)  # 初始化左侧填充
    current_data = torch.cat([x, input_data], dim=3)  # (B, 1, H, W+1)
    current_data = current_data[:, :, :, :W]  # 裁剪到原始宽度
    output_data = current_data * dlr * nlr + input_data * (1 - dlr * nlr)

    # 右到左传播
    x = torch.zeros((B, 1, H, 1), device=input_data.device)  # 初始化右侧填充
    current_data = torch.cat([output_data, x], dim=3)  # (B, 1, H, W+1)
    current_data = current_data[:, :, :, 1:]  # 裁剪到原始宽度
    output_data = current_data * drl * nrl + output_data * (1 - drl * nrl)

    # 上到下传播
    x = torch.zeros((B, 1, 1, W), device=input_data.device)  # 初始化顶部填充
    current_data = torch.cat([x, output_data], dim=2)  # (B, 1, H+1, W)
    current_data = current_data[:, :, :H, :]  # 裁剪到原始高度
    output_data = current_data * dud * nud + output_data * (1 - dud * nud)

    # 下到上传播
    x = torch.zeros((B, 1, 1, W), device=input_data.device)  # 初始化底部填充
    current_data = torch.cat([output_data, x], dim=2)  # (B, 1, H+1, W)
    current_data = current_data[:, :, 1:, :]  # 裁剪到原始高度
    output_data = current_data * ddu * ndu + output_data * (1 - ddu * ndu)

    return output_data




# 边缘感知精炼网络
from .teed.ted import TED
class EdgeAwareRefinement(nn.Module):
    def __init__(self):
        super(EdgeAwareRefinement, self).__init__()

        self.edge_conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)  # 输入通道改为 2（边缘 + 缩放输入）
        self.edge_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.edge_predictor = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.ted = TED().to("cuda")

    def forward(self, inputs, initial_depth,depth_min,depth_max):
        """
        使用边缘特征对深度进行优化。
        Args:
            inputs: 输入张量，形状为 (B, 1, H, W)。
            edges: 边缘图张量，形状为 (B, 1, H, W)。
            initial_depth: 初始深度图，形状为 (B, 1, H, W)。
        Returns:
            refined_depth: 优化后的深度图。
        """
        B, _, H, W = inputs.shape

        # edges = extract_edges(inputs, H, W)
        # print(inputs.shape)
        _,edges = self.ted(inputs)
        # print(len(edges2))

        edge_inputs = torch.cat([edges, inputs], dim=1)  # (B, 2, H, W)

        # 边缘特征提取
        x = F.relu(self.edge_conv1(edge_inputs))
        x = F.relu(self.edge_conv2(x))

        # 预测边缘权重
        edge_weights = self.edge_predictor(x)  # (B, 8, H, W)
        edge_weights = edge_weights + edges.repeat(1, 8, 1, 1)  # 拼接边缘图信息
        edge_weights = torch.clamp(edge_weights, 0.0, 1.0)  # 限制在 [0, 1]

        # 拆分边缘权重
        dlr, drl, dud, ddu, nlr, nrl, nud, ndu = torch.split(edge_weights, 1, dim=1)
        # 传播深度
        refined_depth = propagate(initial_depth, dlr, drl, dud, ddu,nlr, nrl, nud, ndu, dim=1)

        # visualize_tensor(inputs.squeeze(), "img")
        # visualize_tensor(edges.squeeze(), "edges")
        #
        # visualize_tensor(edge_weights.mean(dim=1), "edge_weights (mean across channels)")
        # visualize_tensor(dlr.squeeze(), "dlr (down-left to right)")
        # visualize_tensor(drl.squeeze(), "drl (down-right to left)")
        # visualize_tensor(dud.squeeze(), "dud (down-up to down)")
        # visualize_tensor(ddu.squeeze(), "ddu (down-down to up)")
        # visualize_tensor(nlr.squeeze(), "nlr (normal left to right)")
        # visualize_tensor(nrl.squeeze(), "nrl (normal right to left)")
        # visualize_tensor(nud.squeeze(), "nud (normal up to down)")
        # visualize_tensor(ndu.squeeze(), "ndu (normal down to up)")

        return refined_depth










# def visualize_tensor(tensor, title):
#     # Move the tensor to the CPU before converting it to a NumPy array
#     tensor_cpu = tensor.squeeze().detach().cpu().numpy()
#     plt.imshow(tensor_cpu, cmap='viridis')
#     plt.colorbar()
#     plt.title(title)
#     plt.axis('off')
#     plt.show()
def visualize_tensor(tensor, title):
    # Move the tensor to the CPU before converting it to a NumPy array
    tensor_cpu = tensor.squeeze().detach().cpu().numpy()

    # Handle multi-channel tensors (e.g., 3-channel RGB)
    if tensor_cpu.ndim == 3 and tensor_cpu.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        tensor_cpu = tensor_cpu.transpose(1, 2, 0)

    plt.imshow(tensor_cpu, cmap='viridis' if tensor_cpu.ndim == 2 else None)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()


# Visualizing edge_weights and each component




#
import torchvision.utils as vutils
import os
import matplotlib.cm as cm
from PIL import Image


# 定义归一化函数
def normalize_feature_map(feature_map):
    feature_map_min = feature_map.min()
    feature_map_max = feature_map.max()
    if feature_map_max > feature_map_min:
        return (feature_map - feature_map_min) / (feature_map_max - feature_map_min)
    else:
        return feature_map - feature_map_min


# 定义最大值融合并保存为彩色图的函数
def save_fused_feature_map(features, layer_name):
    # 确保特征图是 CPU 张量
    if features.is_cuda:
        features = features.cpu()

    # 检查并去掉 batch 维度
    if features.dim() == 4 and features.shape[0] == 1:
        features = features.squeeze(0)  # 从 (1, C, H, W) 变为 (C, H, W)
    elif features.dim() != 3:
        raise ValueError(f"Expected features to have 3 dims (C, H, W) or 4 dims with batch=1, got {features.shape}")

    # 沿着通道维度取最大值，得到 (H, W)
    fused_feature = torch.max(features, dim=0)[0]

    # 归一化融合后的特征图
    normalized_feature = normalize_feature_map(fused_feature)

    # 转换为 numpy 数组
    feature_np = normalized_feature.numpy()

    # 使用颜色映射转换为 RGB 图像
    colormap = cm.get_cmap('viridis')  # 可选：'jet', 'plasma', 'magma' 等
    colored_feature = colormap(feature_np)  # 输出形状 (H, W, 4)，RGBA 格式

    # 去掉 Alpha 通道，仅保留 RGB，形状变为 (H, W, 3)
    colored_feature_rgb = colored_feature[:, :, :3]

    # 转换为 0-255 范围的 uint8 类型
    colored_feature_rgb = (colored_feature_rgb * 255).astype(np.uint8)

    # 直接保存为 JPG 图像，不使用 matplotlib
    Image.fromarray(colored_feature_rgb).save(f'{layer_name}.jpg')
    print(layer_name)


def save_fused_feature_map2(features, layer_name):
    # 确保特征图是 CPU 张量
    # if features.is_cuda:
    tensor = features
    min_val = tensor.min()
    max_val = tensor.max()

    # 步骤 2：反归一化到 0-255
    tensor = (tensor - min_val) / (max_val - min_val)  # 归一化到 0-1
    tensor = tensor * 255  # 缩放到 0-255

    # 步骤 3：调整形状并转换为 uint8
    tensor = tensor.squeeze(0)  # 移除 batch 维度，变为 [3, 384, 768]
    tensor = tensor.permute(1, 2, 0)  # 转换为 [384, 768, 3]
    tensor = tensor.clamp(0, 255).byte()  # 限制范围并转换为 uint8

    # 步骤 4：保存为 JPG
    img = Image.fromarray(tensor.numpy(), mode='RGB')
    img.save(os.path.join(f'{layer_name}_normalized_image.jpg'))
    # 保存图像
    # Image.fromarray(tensor).save(f'{layer_name}_normalized_image.jpg')


# step 1. feature extraction


class RefineNet_(nn.Module):
    def __init__(self):
        super(RefineNet_, self).__init__()
        self.edge = EdgeAwareRefinement()
        # self.conv1 = ConvBnReLU(8, 32)
        # self.conv2 = ConvBnReLU(32, 32)
        # self.conv3 = ConvBnReLU(32, 32)
        # self.res = ConvBnReLU(32, 1)
        # self.net = RefineNet("cuda")


    def forward(self, img, depth_init,depth_min,depth_max):

        depth_init = self.edge(img, depth_init,depth_min,depth_max)

        return depth_init




    def sobel_filter(self, image):
        # 保存原始分辨率
        original_resolution = image.shape[2:]

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).to(image.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(
            0).to(image.device)

        if image.dim() == 4:
            if image.size(1) == 1:
                # 单通道图像处理
                grad_x = F.conv2d(image, sobel_x)
                grad_y = F.conv2d(image, sobel_y)
                gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
            else:
                # 多通道图像处理
                channels = torch.split(image, 1, dim=1)
                grad_x = [F.conv2d(channel, sobel_x) for channel in channels]
                grad_y = [F.conv2d(channel, sobel_y) for channel in channels]
                grad_x = torch.cat(grad_x, dim=1)
                grad_y = torch.cat(grad_y, dim=1)
                gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

            # 归一化梯度幅值
            gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (
                        gradient_magnitude.max() - gradient_magnitude.min())

            # 恢复原始分辨率
            gradient_magnitude = F.interpolate(gradient_magnitude, size=original_resolution, mode='bilinear',
                                               align_corners=False)

            return gradient_magnitude
        else:
            raise ValueError("conv2d的输入维度无效")


class RefineNet(nn.Module):
    """ HED network. """

    def __init__(self, device):
        super(RefineNet, self).__init__()
        # Layers.
        self.conv1_1 = nn.Conv2d(5, 64, 3, padding=35)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)

        self.relu = nn.ReLU()
        # Note: ceil_mode – when True, will use ceil instead of floor to compute the output shape.
        #       The reason to use ceil mode here is that later we need to upsample the feature maps and crop the results
        #       in order to have the same shape as the original image. If ceil mode is not used, the up-sampled feature
        #       maps will possibly be smaller than the original images.
        self.maxpool = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.score_dsn1 = nn.Conv2d(64, 1, 1)  # Out channels: 1.
        self.score_dsn2 = nn.Conv2d(128, 1, 1)
        self.score_dsn3 = nn.Conv2d(256, 1, 1)
        self.score_dsn4 = nn.Conv2d(512, 1, 1)
        self.score_dsn5 = nn.Conv2d(512, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)

        # Fixed bilinear weights.
        self.weight_deconv2 = make_bilinear_weights(4, 1).to(device)
        self.weight_deconv3 = make_bilinear_weights(8, 1).to(device)
        self.weight_deconv4 = make_bilinear_weights(16, 1).to(device)
        self.weight_deconv5 = make_bilinear_weights(32, 1).to(device)

        # Prepare for aligned crop.
        self.crop1_margin, self.crop2_margin, self.crop3_margin, self.crop4_margin, self.crop5_margin = \
            self.prepare_aligned_crop()

    # noinspection PyMethodMayBeStatic
    def prepare_aligned_crop(self):
        """ Prepare for aligned crop. """

        # Re-implement the logic in deploy.prototxt and
        #   /hed/src/caffe/layers/crop_layer.cpp of official repo.
        # Other reference materials:
        #   hed/include/caffe/layer.hpp
        #   hed/include/caffe/vision_layers.hpp
        #   hed/include/caffe/util/coords.hpp
        #   https://groups.google.com/forum/#!topic/caffe-users/YSRYy7Nd9J8

        def map_inv(m):
            """ Mapping inverse. """
            a, b = m
            return 1 / a, -b / a

        def map_compose(m1, m2):
            """ Mapping compose. """
            a1, b1 = m1
            a2, b2 = m2
            return a1 * a2, a1 * b2 + b1

        def deconv_map(kernel_h, stride_h, pad_h):
            """ Deconvolution coordinates mapping. """
            return stride_h, (kernel_h - 1) / 2 - pad_h

        def conv_map(kernel_h, stride_h, pad_h):
            """ Convolution coordinates mapping. """
            return map_inv(deconv_map(kernel_h, stride_h, pad_h))

        def pool_map(kernel_h, stride_h, pad_h):
            """ Pooling coordinates mapping. """
            return conv_map(kernel_h, stride_h, pad_h)

        x_map = (1, 0)
        conv1_1_map = map_compose(conv_map(3, 1, 35), x_map)
        conv1_2_map = map_compose(conv_map(3, 1, 1), conv1_1_map)
        pool1_map = map_compose(pool_map(2, 2, 0), conv1_2_map)

        conv2_1_map = map_compose(conv_map(3, 1, 1), pool1_map)
        conv2_2_map = map_compose(conv_map(3, 1, 1), conv2_1_map)
        pool2_map = map_compose(pool_map(2, 2, 0), conv2_2_map)

        conv3_1_map = map_compose(conv_map(3, 1, 1), pool2_map)
        conv3_2_map = map_compose(conv_map(3, 1, 1), conv3_1_map)
        conv3_3_map = map_compose(conv_map(3, 1, 1), conv3_2_map)
        pool3_map = map_compose(pool_map(2, 2, 0), conv3_3_map)

        conv4_1_map = map_compose(conv_map(3, 1, 1), pool3_map)
        conv4_2_map = map_compose(conv_map(3, 1, 1), conv4_1_map)
        conv4_3_map = map_compose(conv_map(3, 1, 1), conv4_2_map)
        pool4_map = map_compose(pool_map(2, 2, 0), conv4_3_map)

        conv5_1_map = map_compose(conv_map(3, 1, 1), pool4_map)
        conv5_2_map = map_compose(conv_map(3, 1, 1), conv5_1_map)
        conv5_3_map = map_compose(conv_map(3, 1, 1), conv5_2_map)

        score_dsn1_map = conv1_2_map
        score_dsn2_map = conv2_2_map
        score_dsn3_map = conv3_3_map
        score_dsn4_map = conv4_3_map
        score_dsn5_map = conv5_3_map

        upsample2_map = map_compose(deconv_map(4, 2, 0), score_dsn2_map)
        upsample3_map = map_compose(deconv_map(8, 4, 0), score_dsn3_map)
        upsample4_map = map_compose(deconv_map(16, 8, 0), score_dsn4_map)
        upsample5_map = map_compose(deconv_map(32, 16, 0), score_dsn5_map)

        crop1_margin = int(score_dsn1_map[1])
        crop2_margin = int(upsample2_map[1])
        crop3_margin = int(upsample3_map[1])
        crop4_margin = int(upsample4_map[1])
        crop5_margin = int(upsample5_map[1])

        return crop1_margin, crop2_margin, crop3_margin, crop4_margin, crop5_margin

    def forward(self, x,y,liner):
        # print(y.shape)
        x = torch.cat((x, y,liner), 1)
        # VGG-16 network.
        image_h, image_w = x.shape[2], x.shape[3]
        conv1_1 = self.relu(self.conv1_1(x))
        conv1_2 = self.relu(self.conv1_2(conv1_1))  # Side output 1.
        pool1 = self.maxpool(conv1_2)

        conv2_1 = self.relu(self.conv2_1(pool1))
        conv2_2 = self.relu(self.conv2_2(conv2_1))  # Side output 2.
        pool2 = self.maxpool(conv2_2)

        conv3_1 = self.relu(self.conv3_1(pool2))
        conv3_2 = self.relu(self.conv3_2(conv3_1))
        conv3_3 = self.relu(self.conv3_3(conv3_2))  # Side output 3.
        pool3 = self.maxpool(conv3_3)

        conv4_1 = self.relu(self.conv4_1(pool3))
        conv4_2 = self.relu(self.conv4_2(conv4_1))
        conv4_3 = self.relu(self.conv4_3(conv4_2))  # Side output 4.
        pool4 = self.maxpool(conv4_3)

        conv5_1 = self.relu(self.conv5_1(pool4))
        conv5_2 = self.relu(self.conv5_2(conv5_1))
        conv5_3 = self.relu(self.conv5_3(conv5_2))  # Side output 5.

        score_dsn1 = self.score_dsn1(conv1_2)
        score_dsn2 = self.score_dsn2(conv2_2)
        score_dsn3 = self.score_dsn3(conv3_3)
        score_dsn4 = self.score_dsn4(conv4_3)
        score_dsn5 = self.score_dsn5(conv5_3)

        upsample2 = torch.nn.functional.conv_transpose2d(score_dsn2, self.weight_deconv2, stride=2)
        upsample3 = torch.nn.functional.conv_transpose2d(score_dsn3, self.weight_deconv3, stride=4)
        upsample4 = torch.nn.functional.conv_transpose2d(score_dsn4, self.weight_deconv4, stride=8)
        upsample5 = torch.nn.functional.conv_transpose2d(score_dsn5, self.weight_deconv5, stride=16)

        # Aligned cropping.
        crop1 = score_dsn1[:, :, self.crop1_margin:self.crop1_margin + image_h,
                self.crop1_margin:self.crop1_margin + image_w]
        crop2 = upsample2[:, :, self.crop2_margin:self.crop2_margin + image_h,
                self.crop2_margin:self.crop2_margin + image_w]
        crop3 = upsample3[:, :, self.crop3_margin:self.crop3_margin + image_h,
                self.crop3_margin:self.crop3_margin + image_w]
        crop4 = upsample4[:, :, self.crop4_margin:self.crop4_margin + image_h,
                self.crop4_margin:self.crop4_margin + image_w]
        crop5 = upsample5[:, :, self.crop5_margin:self.crop5_margin + image_h,
                self.crop5_margin:self.crop5_margin + image_w]

        # Concatenate according to channels.
        fuse_cat = torch.cat((crop1, crop2, crop3, crop4, crop5), dim=1)
        fuse = self.score_final(fuse_cat)  # Shape: [batch_size, 1, image_h, image_w].
        # results = [crop1, crop2, crop3, crop4, crop5, fuse]
        # results = [torch.sigmoid(r) for r in results]
        # print(torch.sigmoid(fuse).shape)

        return torch.sigmoid(fuse) + y
        # return depth_refined

        # return results



def make_bilinear_weights(size, num_channels):
    """ Generate bi-linear interpolation weights as up-sampling filters (following FCN paper). """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False  # Set not trainable.
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w



def simulate_lowlight(img, brightness_factor=0.3, noise_std=0.05, contrast_factor=0.7):
    """
    模拟低光照效果，适用于标准化图像。
    :param img: 输入标准化图像，形状 [batch, channels, height, width]
    :param brightness_factor: 亮度因子，0.0 表示全黑，1.0 表示原亮度
    :param noise_std: 高斯噪声标准差，模拟低光照噪声
    :param contrast_factor: 对比度因子，<1.0 降低对比度
    :return: 变暗后的标准化图像
    """
    # 1. 反标准化：将标准化图像转换回原始像素范围
    # 假设原始图像的均值和方差在标准化时已计算
    # 这里我们需要近似还原（假设原始像素值范围为 [0, 1]）
    # 如果你有原始均值和方差，可以直接使用
    mean = img.mean(dim=(2, 3), keepdim=True)  # [1, 3, 1, 1]
    std = img.std(dim=(2, 3), keepdim=True)    # [1, 3, 1, 1]
    img_orig = img * std + mean  # 反标准化

    # 2. 模拟低光照效果
    # 降低亮度
    img_dark = img_orig * brightness_factor
    # 降低对比度
    img_dark = contrast_factor * (img_dark - img_dark.mean(dim=(2, 3), keepdim=True)) + img_dark.mean(dim=(2, 3), keepdim=True)
    # 添加高斯噪声
    noise = torch.normal(mean=0.0, std=noise_std, size=img_dark.shape, device=img_dark.device)
    img_dark = img_dark + noise
    # 确保像素值非负（模拟真实图像）
    img_dark = torch.clamp(img_dark, min=0.0)

    # 3. 重新标准化：恢复零均值、单位方差
    mean_dark = img_dark.mean(dim=(2, 3), keepdim=True)
    var_dark = img_dark.var(dim=(2, 3), keepdim=True)
    img_dark_std = (img_dark - mean_dark) / (torch.sqrt(var_dark + 1e-8))

    return img_dark_std


import torchvision.transforms as transforms

def tensor_to_pil(tensor):
    tensor = tensor.squeeze(0)  # 去掉批次维度 [3, H, W]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())  # 归一化到 [0, 1]
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)  # 转为 uint8
    return transforms.ToPILImage()(tensor)

def save_combined_image(image1, image2, output_path="output_combined.png"):
    # 将两个输入的 PIL 图片进行横向拼接
    total_width = image1.width + image2.width
    max_height = max(image1.height, image2.height)
    combined = Image.new("RGB", (total_width, max_height))
    
    # 拼接前后图
    combined.paste(image1, (0, 0))
    combined.paste(image2, (image1.width, 0))
    
    # 保存拼接后的图片
    combined.save(output_path)
    print(f"图片已保存: {output_path}")


    

def simulate_non_uniform_low_light(img_batch, method='shadow', device='cuda'):
    """
    模拟非均匀低光照效果，适用于GPU上的批量处理
    参数：
        img_batch: 输入图像批次，形状 (bs, C, H, W)，像素值 [0, 255]，float32
        method: 光照模拟方法 ('shadow')
        device: 计算设备 ('cuda' 或 'cpu')
    返回：
        low_light_img_batch: 模拟后的低光照图像批次，形状 (bs, C, H, W)，float32
    """
    bs, C, H, W = img_batch.shape  # 例如 (bs, 3, 384, 768)
    
    # 初始化掩码为全1，形状 (bs, 1, H, W)
    mask = torch.ones(bs, 1, H, W, device=device, dtype=torch.float32)
    
    # 为每个批次样本随机生成1-3个阴影
    num_shadows = torch.randint(1, 8, (bs,), device=device)
    
    for b in range(bs):
        # for _ in range(num_shadows[b]):
        #     # 随机阴影中心和半径
        #     center_x = torch.randint(0, W, (1,), device=device).item()
        #     center_y = torch.randint(0, H, (1,), device=device).item()
        #     radius = torch.randint(50, 150, (1,), device=device).item()
            
        #     # 创建网格
        #     x = torch.arange(W, device=device, dtype=torch.float32)
        #     y = torch.arange(H, device=device, dtype=torch.float32)
        #     X, Y = torch.meshgrid(y, x, indexing='ij')
            
        #     # 计算高斯形状的阴影
        #     distance = torch.sqrt((X - center_x)**2 + (Y - center_y)**2)
        #     shadow = torch.exp(-distance**2 / (2 * radius**2))  # (H, W)
        #     shadow = shadow.view(1, H, W)  # (1, H, W)
        #     mask[b] *= (1 - 0.5 * shadow)  # 更新该样本的掩码
    
        # mask[b] = np.ones((H, W, 1), dtype=np.float32) * 0.1  # 整体亮度降低到50%
        # noise = np.random.normal(-1, 1, (bs, 3, H, W)).astype(np.float32)

        # brightness_factor = random.uniform(0.1, 0.1)

        brightness_factor = 0.5
        mask = torch.ones((bs, 1, H, W), dtype=torch.float32, device='cuda') * brightness_factor
        noise = torch.randn((bs, C, H, W), dtype=torch.float32, device='cuda')  # 默认均值0，方差1
        noise = noise * 1 + (-1)  # 均值 -1，标准差 1


        # low_light_img = low_light_img + noise 


    # 应用掩码到图像批次，mask 广播到 (bs, C, H, W)
    low_light_img_batch = img_batch * mask + noise
    
    return low_light_img_batch
class CascadeMVSNet(nn.Module):
    def __init__(self, refine=True, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=True,
                 grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(CascadeMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        # self.min_interval = min_interval
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        # 2021-04-20 changed the out_channels of featurenet for share_cr
        self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        if self.share_cr:
            self.cost_regularization = CostRegNet_old(in_channels=128, base_channels=8)  # 16, 8
            # self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=128,
            #                                                      base_channels=self.cr_base_chs[i])
            #                                           for i in range(self.num_stage)])  # [32, 16, 8] [8, 8, 8]




        else:
            print('hhhhhhhhhhhhh')
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=64 + self.feature.out_channels[i],
                                                                 base_channels=self.cr_base_chs[i])
                                                      for i in range(self.num_stage)])   # [32, 16, 8] [8, 8, 8]




        self.DepthNet = DepthNet()

        # print(self.DepthNet)
        if self.refine:
            self.refine_network = RefineNet_()



        # self.diffusion = Diffusion_2d()
        self.diffusion = DiffusionModel_()

        # pretrained_diffusion = torch.load('/opt/data/private/2025code/MVS_lowLT/MVS_lowLT/model_000000_207.4482.ckpt',
        #                map_location='cuda')["model"]


        # model_diffusion = self.diffusion.state_dict()

        # pretrained_diffusion = {k.replace('module.diffusion.', ''): v for k, v in pretrained_diffusion.items() if k.startswith("module.diffusion")}
        # print(".........")
        # # print(pretrained_diffusion)
        # for name, param in self.diffusion.named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        #     # param.requires_grad = False
        # model_diffusion.update(pretrained_diffusion)
        # self.diffusion.load_state_dict(model_diffusion, strict=True)

        # self.diffusion.eval()

    def forward(self, imgs, proj_matrices, depth_values,test):


    # def forward(self, imgs, proj_matrices, depth_values):
        # depth_min = float(depth_values[0, 0].cpu().numpy())
        # depth_max = float(depth_values[0, 1].cpu().numpy())
        # depth_interval = (depth_max - depth_min) / depth_values.size(1)
        # print(depth_min,depth_max,depth_interval)




        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -2].cpu().numpy())
        depth_interval = float(depth_values[0, -1].cpu().numpy())
        depth_range = depth_values[:, 0:-1]







        # print(depth_min,depth_max,depth_interval,depth_range)

        # step 1. feature extraction
        features = []
        deps_features = []
        lossdf = 0


        for nview_idx in range(imgs.size(1)):  #imgs shape (B, N, C, H, W)
            # img = imgs[:, nview_idx]
            # print(img.shape)


            # imgsadd = imgs.clone()

            # imgsadd[:, nview_idx] = simulate_lowlight(imgs[:, nview_idx], brightness_factor=0.3, noise_std=0.05, contrast_factor=0.7)
            # # imgsadd[:, nview_idx] = add_shadows(imgs[:, nview_idx], num_shadows=10, max_shadow_intensity=0.95, base_seed=42)
            # # img = imgsadd[:, nview_idx]

            # image_before = tensor_to_pil(imgsadd[:, nview_idx])

            # imgsadd[:, nview_idx] ,lossdf= self.diffusion(imgsadd[:, nview_idx],imgs[:, nview_idx])
            # img = imgsadd[:, nview_idx]

            # image_after = tensor_to_pil(imgsadd[:, nview_idx]) 



            # save_combined_image(image_before, image_after, "output_combined.png")
            # import time
            # time.sleep(2)

            # if nview_idx == 0:
            #     with torch.no_grad():

            #         img, _ = self.diffusion(imgs[:, nview_idx])
            #     refinimage = img.clone()

            # else:
            #     img = imgs[:, nview_idx]


            input_slice = imgs[:, nview_idx].clone()  
            img_with_shadows = simulate_non_uniform_low_light(input_slice, method='all', device="cuda")

            mean = img_with_shadows.mean(dim=(2, 3), keepdim=True)  # (bs, C, 1, 1)
            var = img_with_shadows.var(dim=(2, 3), keepdim=True)    # (bs, C, 1, 1)
            img_with_shadows = (img_with_shadows - mean) / (torch.sqrt(var) + 1e-8)
            # img = img_with_shadows
            with torch.no_grad():
                img = self.diffusion.sample(img_with_shadows, num_steps=1, use_ddim=True, eta=0.0)

            # img = img_with_shadows
            # img = imgs[:, nview_idx]

            if nview_idx == 0:
                # with torch.no_grad():
                    # enhanced_img = self.diffusion.sample(img_with_shadows, num_steps=1, use_ddim=True, eta=0.0)

                    # img = self.diffusion.sample(imgs[:, nview_idx], num_steps=1, use_ddim=True, eta=0.0)
                    # img = imgs[:, nview_idx]
                    # save_combined_image(tensor_to_pil(imgs[:, nview_idx]), tensor_to_pil(img), "output_combined3.png")

                refinimage = img.clone()


            # else:
            #     img = imgs[:, nview_idx]


                # image_before = tensor_to_pil(imgs[:, nview_idx])

                # image_after = tensor_to_pil(img) 

                # save_combined_image(image_before, image_after, "output_combinedsour.png")

        #     enhanced_img = self.diffusion.sample(lowlight_img, num_steps=10, use_ddim=True, eta=1.0)

                    # img, _ = self.diffusion(imgs[:, nview_idx])

            # if nview_idx == 0:
            # fea,deps = self.feature(nview_idx, img)
            # with torch.no_grad():
            fea,deps = self.feature(nview_idx, img,test)
            features.append(fea)
            deps_features.append(deps)




        outputs = {}
        outputs["lossdf"] = lossdf

        # depth, cur_depth = None, None
        depth, cur_depth, exp_var = None, None, None

        for stage_idx in range(self.num_stage):
            # print("*********************stage{}*********************".format(stage_idx + 1))
            #stage feature, proj_mats, scales
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            deps_feature_stage = [d["stage{}".format(stage_idx + 1)] for d in deps_features]





            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                    exp_var = exp_var.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                                                [img.shape[2], img.shape[3]], mode='bilinear',
                                                align_corners=Align_Corners_Range).squeeze(1)

                exp_var = F.interpolate(exp_var.unsqueeze(1), [img.shape[2], img.shape[3]], mode='bilinear')
            else:
                cur_depth = depth_range
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                                                          exp_var=exp_var,
                                                          ndepth=self.ndepths[stage_idx],
                                                        depth_inteval_pixel=self.depth_interals_ratio[stage_idx]* depth_interval,
                                                        dtype=img[0].dtype,
                                                        device=img[0].device,
                                                        shape=[img.shape[0], img.shape[2], img.shape[3]],
                                                        max_depth=depth_max,
                                                        min_depth=depth_min)
            # print(stage_idx)
            # with torch.no_grad():
            outputs_stage = self.DepthNet(features_stage,deps_feature_stage, proj_matrices_stage,stage_idx,
                                          depth_values=F.interpolate(depth_range_samples.unsqueeze(1),
                                                                     [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear',
                                                                     align_corners=Align_Corners_Range).squeeze(1),
                                          num_depth=self.ndepths[stage_idx],
                                          cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx])



            depth = outputs_stage['depth']
            exp_var = outputs_stage['variance']

            # print(imgs[:, 0].shape)  # 应该是 [N, C, H, W]
            # print(depth.shape)  # 应该是 [N, H, W]，需要扩展
            #
            # outputs_stage = self.refine_network(torch.cat((imgs[:, 0], depth), 1))

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)




        # depth map refinement
        if self.refine:
            # print(imgs[:, 0].shape)  # 应该是 [N, C, H, W]
            # print(depth.shape)  # 应该是 [N, H, W]，需要扩展
            # refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth.unsqueeze(1)), 1))

            # print(imgs[:, 0].shape,self.diffusion(imgs[:, 0]))

            # with torch.no_grad():
                # refined_depth = self.refine_network(refinimage, depth.unsqueeze(1), depth_min, depth_max)   
            refined_depth = self.refine_network(refinimage, depth.unsqueeze(1), depth_min, depth_max)

            # self.visualize_tensors(refined_depth, img, depth_init, liner_dep, liner_img,res)

            outputs["refined_depth"] = refined_depth.squeeze(1)
        else:
            outputs["refined_depth"] = depth

        return outputs

    def draw(self, img):



        features = []
        deps_features = []

        nview_idx = 1
        print(img.shape)
        fea, deps = self.feature(nview_idx, img)
        features.append(fea)
        deps_features.append(deps)




            # if nview_idx==2:

            #     save_img=(img.cpu())
        bottom_features = fea["stage3"].cpu()
        mid_features = fea["stage2"].cpu() #16
        top_features = fea["stage1"].cpu() #32
        save_fused_feature_map(bottom_features,"bottom_features")
        save_fused_feature_map(mid_features,"mid_features")
        save_fused_feature_map(top_features,"top_features")

        outputs = {}
        print("1111")
        return outputs
