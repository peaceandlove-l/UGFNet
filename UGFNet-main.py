import torch
import torch.nn as nn
import torch.nn.functional as F
# from deerflow
import math
from typing import List, Optional

class DoRA(nn.Module):
    """
    Basic DoRA module: Weight-Decomposed Low-Rank Adaptation.
    Decomposes pretrained weight W into magnitude m and direction v (unit vector).
    Applies LoRA to the direction: Δv = B A / ||B A||, then W' = (m + Δm) * (v + Δv)
    Here, Δm is a learned scalar/vector, but for simplicity, we use a scalar per module.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int, bias: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank

        # Pretrained weight (frozen, but for adapter, we assume it's external)
        # Here we initialize randomly for standalone, but in practice, decompose from pretrained.
        self.W = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)

        # Decompose: magnitude m (column-wise norm), direction v = W / m
        self.m = nn.Parameter(self.W.norm(dim=0, keepdim=True), requires_grad=False)  # [1, in_dim]
        self.v = nn.Parameter(self.W / self.m.clamp(min=1e-6), requires_grad=False)  # [out_dim, in_dim]

        # LoRA for direction
        self.A = nn.Parameter(torch.randn(rank, in_dim))  # Low-rank down
        self.B = nn.Parameter(torch.randn(out_dim, rank))  # Low-rank up

        # Learned magnitude adjustment (scalar for simplicity)
        self.delta_m = nn.Parameter(torch.zeros(1))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LoRA delta for direction
        delta_v = self.B @ self.A  # [out_dim, in_dim]
        delta_v_norm = delta_v.norm(dim=0, keepdim=True).clamp(min=1e-6)  # [1, in_dim]
        delta_v = delta_v / delta_v_norm  # Unit vector

        # Updated direction
        updated_v = self.v + delta_v

        # Updated magnitude
        updated_m = self.m + self.delta_m

        # Recompose
        updated_W = updated_m * updated_v  # [out_dim, in_dim]

        out = x @ updated_W.T
        if self.bias is not None:
            out += self.bias
        return out


class ExpertDoRA(nn.Module):
    """
    Expert DoRA specialized for different frequencies/scales.
    We simulate focus on frequencies/scales by applying a simple bandpass filter in freq domain.
    Different experts have different frequency bands (low, mid, high for example).
    Adjusted thresholds for better indoor scene capture: low for large structures, high for details.
    """
    def __init__(self, in_dim: int, out_dim: int, rank: int, freq_band: str = 'low'):
        super().__init__()
        self.dora = DoRA(in_dim, out_dim, rank)
        self.freq_band = freq_band  # 'low', 'mid', 'high' for different scales/freq

    def apply_freq_filter(self, x: torch.Tensor) -> torch.Tensor:
        # Simple freq domain filter (assuming x is flattened, but for demo; in practice, adapt for 2D/3D)
        # For semantic seg, x might be [B, HW, C], treat as 1D signal per channel.
        fft_x = torch.fft.fft(x, dim=1)  # FFT along sequence dim

        freqs = torch.fft.fftfreq(x.size(1), device=x.device)
        if self.freq_band == 'low':
            mask = (torch.abs(freqs) < 0.05).float().unsqueeze(0).unsqueeze(
                -1)  # Tighter low for large indoor structures
        elif self.freq_band == 'mid':
            mask = ((torch.abs(freqs) >= 0.05) & (torch.abs(freqs) < 0.25)).float().unsqueeze(0).unsqueeze(-1)
        else:  # high
            mask = (torch.abs(freqs) >= 0.25).float().unsqueeze(0).unsqueeze(-1)  # Broader high for indoor details

        filtered_fft = fft_x * mask
        filtered_x = torch.fft.ifft(filtered_fft, dim=1).real
        return filtered_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        filtered_x = self.apply_freq_filter(x)
        return self.dora(filtered_x)


class SimpleRouter(nn.Module):
    """
    Simplified router: Single-level routing for efficiency.
    Replaces hierarchical for reduced complexity while maintaining dynamic selection.
    """
    def __init__(self, in_dim: int, num_experts: int):
        super().__init__()
        self.router = nn.Linear(in_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is averaged or pooled for routing (e.g., mean over seq)
        routing_input = x.mean(dim=1)  # [B, C]

        logits = self.router(routing_input)  # [B, num_experts]
        probs = F.softmax(logits, dim=-1)
        return probs  # [B, num_experts]


class DinoV3DoRAAdapter(nn.Module):
    """
    Simplified DINOv3-based DoRA Adapter for Semantic Segmentation.
    Core: Dynamic Routing + Multi-Experts + Inter-Layer Sharing + DoRA
    - Multi-Experts: Reduced to 3 experts per segment, each focusing on different freq/scales.
    - Segmented Sharing: Model layers divided into segments, each segment shares a set of experts.
    - Simplified Dynamic Routing: Per segment, a shared single-level router selects experts dynamically.
    - Inter-Layer Sharing: Routers and experts shared within segments.
    Assuming DINOv3 has L layers, divided into S segments.
    For simplicity, assume adapter augments each Transformer layer in residual manner.
    """
    def __init__(self, dino_layers: int, in_dim: int, out_dim: int, rank: int, num_segments: int = 3,
                 num_experts_per_segment: int = 3):
        super().__init__()

        self.num_segments = num_segments
        self.segment_size = dino_layers // num_segments

        # Shared experts per segment: each with different freq bands
        freq_bands = ['low', 'mid', 'high'][:num_experts_per_segment]  # Reduced to 3 for efficiency
        self.segment_experts = nn.ModuleList([
            nn.ModuleList([ExpertDoRA(in_dim, out_dim, rank, freq_band=freq_bands[i % len(freq_bands)])
                           for i in range(num_experts_per_segment)])
            for _ in range(num_segments)
        ])

        # Shared simplified routers per segment
        self.segment_routers = nn.ModuleList([
            SimpleRouter(in_dim, num_experts_per_segment)
            for _ in range(num_segments)
        ])

    def forward(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Determine segment
        segment_idx = min(layer_idx // self.segment_size, self.num_segments - 1)

        router = self.segment_routers[segment_idx]
        experts = self.segment_experts[segment_idx]

        # Get routing probs [B, num_experts]
        probs = router(x)

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in experts], dim=0)  # [num_experts, B, seq, out_dim]

        # Weighted sum: auto select via soft routing (for differentiability)
        # probs [B, num_experts], expert_outputs [num_experts, B, seq, out_dim]
        weighted = torch.einsum('be,ebso -> bso', probs, expert_outputs)

        return weighted


def apply_dora_adapter_to_vit(
        model,
        rank: int = 8,
        num_segments: int = 3,
        num_experts_per_segment: int = 3,
        target_locations: list[str] = ["after_block"],  # e.g., 'after_block', 'after_attn', 'after_mlp'
        modules_to_ignore: list[str] = ["cls_token", "pos_embed", "patch_embed", "head"],
):
    """
    将 DoRA Adapter 应用到 ViT (如 DINOv3) 的指定位置。

    参数:
        model:                  Vision Transformer 模型（nn.Module）
        rank:                   DoRA 的秩
        num_segments:           层分段数
        num_experts_per_segment: 每个分段的专家数
        target_locations:       要插入适配器的位置（e.g., 'after_block' 表示每个 block 后）
        modules_to_ignore:      忽略的模块名

    返回:
        修改后的 model（in-place 修改）
    """
    # 假设 model 有 .blocks 作为 nn.ModuleList of Transformer blocks
    if not hasattr(model, 'blocks') or not isinstance(model.blocks, nn.ModuleList):
        raise ValueError("Model must have 'blocks' as nn.ModuleList (ViT-style).")

    num_layers = len(model.blocks)
    hidden_dim = model.embed_dim  # 假设 DINOv3 有 embed_dim

    # 创建共享适配器
    adapter = DinoV3DoRAAdapter(
        dino_layers=num_layers,
        in_dim=hidden_dim,
        out_dim=hidden_dim,
        rank=rank,
        num_segments=num_segments,
        num_experts_per_segment=num_experts_per_segment,
    )

    # 冻结原模型参数，只训练适配器
    for param in model.parameters():
        param.requires_grad = False
    for param in adapter.parameters():
        param.requires_grad = True

    # 修改 forward 以插入适配器
    original_forward = model.forward

    def adapted_forward(self, x, *args, **kwargs):
        # 假设 forward 是 x = self.patch_embed(x); x = x + self.pos_embed; ...
        # 然后 for block in self.blocks: x = block(x)
        # 这里我们假设简单 ViT forward，实际根据 DINOv3 调整
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            x = block(x)
            if "after_block" in target_locations:
                x = x + adapter(x, i)  # Residual add adapter output

        x = self.norm(x)
        # 对于语义分割，返回特征而不是调用 original_forward 以避免循环
        return x  # 假设语义分割头将使用此特征；根据实际调整

    model.forward = adapted_forward.__get__(model)

    # 添加 adapter 到 model 以保存/加载
    model.adapter = adapter

    print(
        f"Applied DoRA Adapter to {num_layers} layers with {num_segments} segments, {num_experts_per_segment} experts each.")

    return model

# ===== 整体模型（添加 LoRA 到整个 DINO backbone） =====
class Dino(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True):
        super().__init__()
        REPO_DIR = '/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3'
        dinov3_backbone = torch.hub.load(
            REPO_DIR, 'dinov3_vitb16',
            source='local',
            weights='/data/Lsy/sam/lsy/mymodelnew/toolbox/Mymodels/DINO/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
        )
        self.backbone = dinov3_backbone
        backbone_dim = 768
        # 新增：应用 LoRA 到整个 backbone（移除简单冻结）

        apply_dora_adapter_to_vit(self.backbone)

        # if freeze_backbone:
        #     for param in self.backbone.parameters():
        #         param.requires_grad = False
        #     print(f"✓ Backbone已冻结")

        self.U3TR = U3TR(dim=backbone_dim)
        self.UGTD = UGTD(dim=backbone_dim)

        self.up16 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.final = nn.Conv2d(backbone_dim, num_classes,1)

        # self.final2 = nn.Conv2d(24,num_classes,1)

    def forward(self, x, depth):
        B,_,H,W = x.shape

        # 1️ 使用带 LoRA 的 backbone 提取特征（移除 no_grad，因为现在可训练 LoRA）
        r = self.backbone.forward_features(x)
        r = r['x_norm_patchtokens']
        B, N, C = r.shape
        r = r.reshape(B, H//16, W//16, C).permute(0, 3, 1, 2)

        d = self.backbone.forward_features(depth)
        d = d['x_norm_patchtokens']
        B, N, C = d.shape
        d = d.reshape(B, H // 16, W // 16, C).permute(0, 3, 1, 2)
        # print('d',d.shape)#####([2, 768, 30, 40])


        fuse1,_,_ = self.U3TR(r,d)
        fuse2 = self.UGTD(r,d)
        # final = self.final(self.up16(fuse))
        final = self.up16(fuse1 + fuse2)
        final = self.final(final)

        return final


# 测试代码
if __name__ == '__main__':
    print("="*60)
    print("测试 Dino")
    print("="*60)

    # 创建模型
    model = Dino(num_classes=41)

    # 测试前向传播
    x = torch.randn(2, 3, 480, 640)

    print(f"\n输入: {x.shape}")

    # cls_pred, reg_pred = model(x)
    cls_pred = model(x,x)
    print(f"输出:")
    print(f"  Cls pred: {cls_pred.shape}")
    # print(f"  Reg pred: {reg_pred.shape}")

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

    print(f"\n参数统计:")
    print(f"  总参数: {total_params:.2f}M")
    print(f"  可训练参数: {trainable_params:.2f}M")

    print("\n✅ 测试通过!")