import torch
import torch.nn as nn
import torch.nn.functional as F
# from deerflow
import math
#####Uncertainty-guided Geometric Topology Diffusion (UGTD)
class UGTD(nn.Module):
    """
    核心思想：
    1. 利用 uncertainty 构建 topology graph
    2. uncertainty 越高，扩散范围越大
    3. 利用 token graph diffusion 强化全局几何一致性
    4. 与 U3TR 并行形成：
            Local Reconstruction + Global Topology Reasoning

    输入:
        feat_rgb   : [B,C,H,W]
        feat_depth : [B,C,H,W]

    输出:
        topo_feat  : [B,C,H,W]
        topo_loss  : scalar
        uncertainty_map : [B,1,H,W]
    """

    def __init__(self,dim,hidden_dim=512,num_iters=3,k_neighbors=8,temperature=0.07):
        super(UGTD, self).__init__()

        self.dim = dim
        self.num_iters = num_iters
        self.k_neighbors = k_neighbors
        self.temperature = temperature

        # ===== RGB-D Fusion =====
        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # ===== Uncertainty Estimation =====
        self.uncertainty_head = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()
        )

        # ===== Topology Projection =====
        self.topology_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # ===== Diffusion Update =====
        self.diffusion_update = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # ===== Final Refinement =====
        self.refine = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def build_graph(self, tokens, uncertainty):
        """
        构建 uncertainty-aware topology graph

        tokens      : [B,N,C]
        uncertainty : [B,N,1]

        return:
            affinity : [B,N,N]
        """

        B, N, C = tokens.shape

        # ===== Feature Distance =====
        feat_norm = F.normalize(tokens, dim=-1)

        sim = torch.matmul(
            feat_norm,
            feat_norm.transpose(1, 2)
        )  # [B,N,N]

        # ===== uncertainty scaling =====
        sigma = uncertainty + 1e-6

        sigma_matrix = torch.matmul(
            sigma,
            sigma.transpose(1, 2)
        )  # [B,N,N]

        # ===== uncertainty-guided affinity =====
        affinity = torch.exp(
            sim / (self.temperature * sigma_matrix)
        )

        # ===== Top-K Sparsification =====
        k = min(self.k_neighbors, N)

        topk_val, topk_idx = torch.topk(
            affinity,
            k=k,
            dim=-1
        )

        mask = torch.zeros_like(affinity)
        mask.scatter_(-1, topk_idx, 1.0)

        affinity = affinity * mask

        # ===== Row Normalization =====
        affinity = affinity / (
                affinity.sum(dim=-1, keepdim=True) + 1e-6
        )

        return affinity

    def graph_diffusion(self, tokens, affinity):
        """
        Graph Topology Diffusion

        T^{k+1} = A T^k
        """
        x = tokens

        for _ in range(self.num_iters):
            # topology propagation
            propagated = torch.matmul(
                affinity,
                x
            )  # [B,N,C]

            # residual diffusion update
            x = x + self.diffusion_update(propagated)
        return x

    def topology_consistency_loss(self,tokens,propagated,affinity):
        """
        几何拓扑一致性约束

        encourage:
            connected nodes -> similar topology embedding
        """
        diff = propagated.unsqueeze(2) - propagated.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1)
        loss = (affinity * dist).mean()
        return loss

    def forward(self, feat_rgb, feat_depth):

        B, C, H, W = feat_rgb.shape
        N = H * W

        # =====================================================
        # Step1: Flatten Tokens
        # =====================================================

        rgb_tokens = feat_rgb.flatten(2).permute(0, 2, 1)
        depth_tokens = feat_depth.flatten(2).permute(0, 2, 1)

        # =====================================================
        # Step2: RGB-D Fusion
        # =====================================================

        fused = torch.cat(
            [rgb_tokens, depth_tokens],
            dim=-1
        )

        fused_tokens = self.fusion(fused)

        # =====================================================
        # Step3: Uncertainty Estimation
        # =====================================================

        uncertainty = self.uncertainty_head(
            fused_tokens
        )  # [B,N,1]

        # uncertainty-aware enhancement
        weight = torch.sigmoid(uncertainty)

        topo_tokens = fused_tokens * (1 + weight)

        # =====================================================
        # Step4: Topology Embedding
        # =====================================================
        topo_embed = self.topology_proj(
            topo_tokens
        )
        # =====================================================
        # Step5: Build Topology Graph
        # =====================================================
        affinity = self.build_graph(
            topo_embed,
            uncertainty
        )

        # =====================================================
        # Step6: Graph Diffusion
        # =====================================================

        propagated = self.graph_diffusion(
            topo_embed,
            affinity
        )

        # =====================================================
        # Step7: Final Refinement
        # =====================================================

        refined = self.refine(propagated)

        # residual connection
        refined = refined + fused_tokens

        # =====================================================
        # Step8: Topology Consistency Loss
        # =====================================================

        # topo_loss = self.topology_consistency_loss(
        #     topo_embed,
        #     propagated,
        #     affinity
        # )

        # =====================================================
        # Step9: Reshape Back
        # =====================================================

        topo_feat = refined.permute(
            0, 2, 1
        ).reshape(B, C, H, W)
        uncertainty_map = uncertainty.permute(
            0, 2, 1
        ).reshape(B, 1, H, W)

        # return topo_feat, topo_loss, uncertainty_map
        return topo_feat