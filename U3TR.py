import torch
import torch.nn as nn
import torch.nn.functional as F
# from deerflow
import math

###Uncertainty-guided 3D Token Reconstruction (U3TR)
class U3TR(nn.Module):
    def __init__(self, dim, hidden_dim=512):
        super(U3TR, self).__init__()

        # 1. RGB-D fusion (pseudo 3D token)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # 2. Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # 保证 U >= 0
        )

        # 3. Reconstruction decoder
        self.reconstruction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.refine_block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, feat_rgb, feat_depth):
        """
        feat_rgb: [B, C, H, W]
        feat_depth: [B, C, H, W]
        """

        B, C, H, W = feat_rgb.shape
        N = H * W

        # ===== Step 1: flatten to tokens =====
        rgb_tokens = feat_rgb.flatten(2).permute(0, 2, 1)   # [B, N, C]
        depth_tokens = feat_depth.flatten(2).permute(0, 2, 1)

        # ===== Step 2: fusion (pseudo 3D tokens) =====
        fused_tokens = torch.cat([rgb_tokens, depth_tokens], dim=-1)  # [B, N, 2C]
        t_3d = self.fusion(fused_tokens)  # [B, N, C]

        # ===== Step 3: uncertainty estimation =====
        U = self.uncertainty_head(t_3d)  # [B, N, 1]

        attn_weight = torch.sigmoid(U)  # [B,N,1]不确定性引导 attention
        t_3d = t_3d * (1 + attn_weight)

        # ===== Step 3.5: Top-K token refinement =====
        U_score = U.squeeze(-1)  # [B,N]
        k = int(0.3 * U_score.shape[1])

        _, topk_idx = torch.topk(U_score, k, dim=1)

        mask = torch.zeros_like(U_score)
        mask.scatter_(1, topk_idx, 1.0)
        mask = mask.unsqueeze(-1)  # [B,N,1]

        # ===== Step 3.6: refinement (核心！！) =====
        refined_tokens = t_3d * (1 + mask)  # 强化困难token
        # 或更强一点：
        # refined_tokens = t_3d + mask * self.refine_mlp(t_3d)
        # refined_tokens = t_3d + mask * self.refine_block(t_3d)


        # ===== Step 4: reconstruction =====
        t_hat = self.reconstruction(refined_tokens)  # [B, N, C]

        # ===== Step 5: reconstruction loss =====
        # 原始 token（可以选 rgb 或 fused）
        t_orig = rgb_tokens  # 推荐用 RGB 主分支

        # 权重 w = (1 - exp(-U))
        weight = 1 - torch.exp(-U)

        rec_loss = (weight * (t_orig - t_hat) ** 2).mean()

        # reshape back (optional)
        t_3d_map = t_3d.permute(0, 2, 1).reshape(B, C, H, W)

        t_hat = t_hat.permute(0, 2, 1).reshape(B, C, H, W)

        return t_3d_map, rec_loss, t_hat
