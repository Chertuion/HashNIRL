import torch
import torch.nn.functional as F
import torch.nn as nn

def random_mask(image, mask_prob=0.2):
    mask = torch.zeros_like(image)
    mask_prob_tensor = torch.tensor(mask_prob, device=image.device)
    mask = torch.bernoulli(mask_prob_tensor.expand_as(mask))

    return mask

class InfoNCELoss(nn.Module):
    def __init__(self, temperature, num_negative_samples=100):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature
        self.num_negative_samples = num_negative_samples

    def forward(self, zi, zj):
        batch_size = zi.size(0)
        
        zi_norm = torch.norm(zi, p=2, dim=1, keepdim=True) ** 2
        zj_norm = torch.norm(zj, p=2, dim=1, keepdim=True) ** 2

        d_ij_aug_diag = (zi_norm + zj_norm - 2 * (zi * zj).sum(dim=1, keepdim=True)) / (2 * self.temperature)
        d_ij = (zi_norm + zi_norm.T - 2 * torch.matmul(zi, zi.T)) / (2 * self.temperature)

        pos_mask = torch.eye(batch_size, dtype=torch.bool, device=d_ij.device)

        neg_mask = ~pos_mask

        d_ij_neg = d_ij[neg_mask].view(batch_size, -1)
        topk_neg_distances, _ = torch.topk(d_ij_neg, self.num_negative_samples, dim=1, largest=True)

        exp_pos = torch.exp(-d_ij_aug_diag).view(-1)
        exp_neg = torch.exp(-topk_neg_distances).sum(dim=1)

        loss = -torch.log(exp_pos / (exp_neg + exp_pos + 1e-5)).mean()
        return loss

# def compute_div_penalty(proj, featurizer, ready_features, image, temp, mask_p):
    
#     image = image.view(image.shape[0], 2 * 14 * 14)
#     aug_image = image * random_mask(image)
    
#     zi = proj(ready_features)
#     zj = proj(featurizer(aug_image))

#     mask = int(zi.size(1) * mask_p)
#     zi[:, :mask] = 0
#     zj[:, :mask] = 0
    
#     num_negative_samples = int(0.7 * image.shape[0]) if int(0.7 * image.shape[0] <= 100) else 100
#     loss = InfoNCELoss(temperature=temp, num_negative_samples=num_negative_samples)(zi, zj)
#     return loss

def compute_div_penalty(proj, featurizer, ready_features, spectrum, temp, mask_p):
    """
    proj: 投影头（mlp.proj）
    featurizer: 特征提取主干（mlp._main 或 model._main）
    ready_features: 原始 features，来自主干
    spectrum: 原始光谱数据，形状为 [B, L]
    temp: InfoNCE 的温度参数
    mask_p: 遮罩比例 (0~1)，用于构造不同视图
    """

    def random_mask(x, mask_p):
        x_aug = x.clone()
        if x.dim() == 1:
            # [L] -> 随机 mask 某些维度
            num_mask = int(x.size(0) * mask_p)
            idx = torch.randperm(x.size(0))[:num_mask]
            x_aug[idx] = 0
        elif x.dim() == 2:
            # [B, L] -> 每条光谱 mask 同一维度
            num_mask = int(x.size(1) * mask_p)
            idx = torch.randperm(x.size(1))[:num_mask]
            x_aug[:, idx] = 0
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        return x_aug

    spectrum = spectrum.float()
    # 创建增强样本
    aug_spectrum = random_mask(spectrum, mask_p)

    # 得到投影特征
    zi = proj(ready_features)                # anchor
    aug_spectrum = aug_spectrum.float().unsqueeze(1)
    zj = proj(featurizer(aug_spectrum))      # positive

    # 重新掩码（可选：再次置零一部分维度）
    mask_dim = int(zi.size(1) * mask_p)
    zi[:, :mask_dim] = 0
    zj[:, :mask_dim] = 0

    # 设定 InfoNCE 参数
    num_negative_samples = min(int(0.7 * spectrum.shape[0]), 100)

    # 计算 InfoNCE 损失
    loss = InfoNCELoss(temperature=temp, num_negative_samples=num_negative_samples)(zi, zj)
    return loss


