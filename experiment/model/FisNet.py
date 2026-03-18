"""
FIS-Net: Multimodal FIS Assessment Network
==========================================
基于 Mamba SSM 的多模态咨询师促进性人际技能 (FIS) 评估模型。

架构总览（自底向上）:
  1. 单模态编码器（参数独立，共享 Mamba 基础架构）
     - VisualEncoder:  Subfeature Grouping Attention → Mamba
     - AudioEncoder:   Wav2Vec ⊕ Librosa 双流融合 → Mamba
     - TextEncoder:    BERT embedding → Mamba → token-to-word pooling
  2. 词级跨模态融合 (Stage 1)
     - GatedCrossModalFusion: Text←Visual / Text←Audio 跨注意力 + 三路门控
  3. 序列级全局融合 (Stage 2)
     - CoupledMambaFusion: 三模态隐状态耦合的 Mamba 扫描
  4. Attentive Pooling → MLP 预测头

设计原则:
  - 所有子模块可独立开关，方便 ablation study
  - forward 接收 collate_fis_batch 输出的 batch dict，无需外部适配
  - 超参数通过 dataclass 配置注入，不硬编码
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


# ============================================================================
#  配置
# ============================================================================

@dataclass
class FISNetConfig:
    """FIS-Net 全局超参数。通过 dataclass 集中管理，方便 ablation 覆盖。"""

    # ---- 模型维度 ----
    d_model: int = 256
    n_labels: int = 9        # FIS 标签维度（abc, empathy, ... 共 9 个连续分数）

    # ---- 输入特征维度（需与 FEATURE_FORMAT / dataloader 一致）----
    video_dim: int = 235     # OpenFace3 混合特征 D_vid（D_au + 196 + 2 + 1 + 1）
    audio_wav2vec_dim: int = 768
    audio_librosa_dim: int = 93
    text_dim: int = 1024     # BERT-large token embedding

    # ---- 视觉子特征分组（按 FEATURE_FORMAT 固定顺序切分）----
    #   video_openface3 = [AU | landmarks(196) | gaze(2) | confidence(1) | time(1)]
    #   D_au = video_dim - 196 - 2 - 1 - 1
    visual_landmark_dim: int = 196
    visual_gaze_dim: int = 2
    visual_meta_dim: int = 2     # confidence + time
    d_group: int = 128           # 分组注意力统一投影维度
    group_attn_heads: int = 4

    # ---- 音频编码器 ----
    d_audio: int = 256           # 双流投影中间维度
    audio_fusion_mode: str = "concat"   # "concat" | "gate"

    # ---- Mamba 编码器 ----
    mamba_n_layers: int = 4
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2

    # ---- 跨模态融合 ----
    cross_attn_heads: int = 4
    cross_attn_dropout: float = 0.1

    # ---- Coupled Mamba 融合 ----
    coupled_n_layers: int = 2

    # ---- Attentive Pooling ----
    pool_hidden: int = 128

    # ---- 预测头 ----
    head_hidden: int = 256
    head_dropout: float = 0.2

    # ---- 任务开关 ----
    task: int = 1                    # 1=仅咨询师, 2=咨询师+来访者交互

    # ---- 任务二：交互模块超参数 ----
    interaction_n_layers: int = 2    # (H) 因果 Transformer 层数
    interaction_n_heads: int = 4
    interaction_dropout: float = 0.1
    d_sync: int = 64                 # (I) 同步性行为投影维度
    sync_n_align: int = 64           # (I) 时间重采样统一长度
    sync_window: int = 8             # (I) 滑动窗口大小
    sync_stride: int = 4             # (I) 滑动窗口步长
    share_encoders: bool = False     # 咨询师/来访者编码器是否共享权重

    # ---- Ablation 开关 ----
    use_visual: bool = True
    use_audio: bool = True
    use_text: bool = True
    use_stage1_fusion: bool = True    # 词级跨模态融合（仅当 fusion_mode="stage1_stage2" 时生效）
    use_stage2_fusion: bool = True    # Coupled Mamba 序列级融合（仅当 fusion_mode="stage1_stage2" 时生效）
    use_grouping_attn: bool = True    # 视觉子特征分组注意力（关闭则直接投影+Mamba）
    use_audio_gate: bool = False      # 音频门控融合（覆盖 audio_fusion_mode）
    use_interaction_attn: bool = True   # (H) 词级交互注意力
    use_synchrony: bool = True          # (I) 多模态行为同步性追踪
    use_alliance: bool = True           # (J) 治疗联盟动态建模
    # ---- 简化融合与文本编码（实验用）----
    fusion_mode: str = "simple_cross"   # "simple_cross": 仅跨模态注意力，无 Stage1/Stage2；"stage1_stage2": 原 GatedCrossModal + CoupledMamba
    text_use_mamba: bool = False        # False 时文本仅 Linear 投影 + token-to-word 池化，无 Mamba

    @property
    def visual_au_dim(self) -> int:
        return self.video_dim - self.visual_landmark_dim - self.visual_gaze_dim - self.visual_meta_dim


# ============================================================================
#  基础组件
# ============================================================================

class MambaBlock(nn.Module):
    """单层 Mamba SSM + 残差 + LayerNorm（Pre-Norm）。"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.mamba(self.norm(x))


class MambaEncoder(nn.Module):
    """L 层堆叠的 Mamba 编码器：Linear Projection → MambaBlock × L → LayerNorm。

    Input:  [B, N, d_in]
    Output: [B, N, d_model]
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_in]
        h = self.proj(x)                    # [B, N, d_model]
        for layer in self.layers:
            h = layer(h)                     # [B, N, d_model]
        return self.norm(h)                  # [B, N, d_model]


class AttentivePooling(nn.Module):
    """学习式注意力池化：将变长序列 [B, N, D] 聚合为 [B, D]。
    score_i = v^T tanh(W h_i + b)，α = softmax(scores, mask)，output = Σ α_i h_i
    """

    def __init__(self, d_in: int, d_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, N, D], mask: [B, N] True=有效
        scores = self.attn(x).squeeze(-1)    # [B, N]
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(scores, dim=-1)    # [B, N]
        # 当某行 mask 全为 False 时 softmax(-inf,...,-inf)=nan，导致输出 nan；置为 0 使该样本输出零向量
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0)
        return (alpha.unsqueeze(-1) * x).sum(dim=1)  # [B, D]


# ============================================================================
#  (B) Visual Subfeature Grouping Attention + Visual Encoder
# ============================================================================

class SubfeatureGroupingAttention(nn.Module):
    """视觉子特征分组注意力。

    将 OpenFace3 混合特征按语义切分为 AU / landmarks / gaze 三组，
    各组独立投影后通过多头自注意力学习组间交互，再用自适应权重加权融合。

    Input:  video_openface3 [B, N_word, D_vid]
    Output: [B, N_word, d_group]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        self.cfg = cfg
        d_au = cfg.visual_au_dim
        d_group = cfg.d_group

        self.au_proj = nn.Linear(d_au, d_group)
        self.landmark_proj = nn.Linear(cfg.visual_landmark_dim, d_group)
        self.gaze_proj = nn.Linear(cfg.visual_gaze_dim, d_group)

        self.group_attn = nn.MultiheadAttention(
            embed_dim=d_group,
            num_heads=cfg.group_attn_heads,
            batch_first=True,
        )
        self.group_norm = nn.LayerNorm(d_group)

        # 自适应组权重：输入为每组的均值，输出为 3 个标量权重
        self.weight_fc = nn.Linear(d_group, 1)

    def forward(self, video_feat: torch.Tensor) -> torch.Tensor:
        # video_feat: [B, N, D_vid]
        B, N, _ = video_feat.shape
        d_au = self.cfg.visual_au_dim
        d_lm = self.cfg.visual_landmark_dim
        d_gz = self.cfg.visual_gaze_dim

        # 按固定顺序切分子特征组：[AU | landmarks | gaze | meta]
        au = video_feat[:, :, :d_au]                                   # [B, N, D_au]
        lm = video_feat[:, :, d_au : d_au + d_lm]                     # [B, N, 196]
        gz = video_feat[:, :, d_au + d_lm : d_au + d_lm + d_gz]       # [B, N, 2]
        # meta (confidence + time) 不参与分组注意力

        au_h = self.au_proj(au)        # [B, N, d_group]
        lm_h = self.landmark_proj(lm)  # [B, N, d_group]
        gz_h = self.gaze_proj(gz)      # [B, N, d_group]

        # 将三组 reshape 为 token 维度做自注意力
        # [B*N, 3, d_group] — 每个时间步有 3 个 "group token"
        group_tokens = torch.stack([au_h, lm_h, gz_h], dim=2)  # [B, N, 3, d_group]
        group_tokens = group_tokens.view(B * N, 3, self.cfg.d_group)

        attn_out, _ = self.group_attn(group_tokens, group_tokens, group_tokens)
        attn_out = self.group_norm(attn_out + group_tokens)     # [B*N, 3, d_group]

        # 自适应组权重
        weights = self.weight_fc(attn_out).squeeze(-1)          # [B*N, 3]
        weights = F.softmax(weights, dim=-1)                    # [B*N, 3]

        fused = (weights.unsqueeze(-1) * attn_out).sum(dim=1)   # [B*N, d_group]
        return fused.view(B, N, self.cfg.d_group)               # [B, N, d_group]


class VisualEncoder(nn.Module):
    """视觉编码器：可选 Subfeature Grouping Attention → Mamba Encoder。

    Input:  video_openface3 [B, N_word, D_vid]
    Output: [B, N_word, d_model]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        self.use_grouping = cfg.use_grouping_attn

        if self.use_grouping:
            self.grouping = SubfeatureGroupingAttention(cfg)
            mamba_in = cfg.d_group
        else:
            mamba_in = cfg.video_dim

        self.mamba_enc = MambaEncoder(
            d_in=mamba_in,
            d_model=cfg.d_model,
            n_layers=cfg.mamba_n_layers,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )

    def forward(self, video_feat: torch.Tensor) -> torch.Tensor:
        # video_feat: [B, N_word, D_vid]
        if self.use_grouping:
            h = self.grouping(video_feat)     # [B, N_word, d_group]
        else:
            h = video_feat                    # [B, N_word, D_vid]
        return self.mamba_enc(h)              # [B, N_word, d_model]


# ============================================================================
#  (C) Audio Dual-Stream Encoder
# ============================================================================

class AudioEncoder(nn.Module):
    """音频双流编码器：Wav2Vec ⊕ Librosa → 投影融合 → Mamba Encoder。

    融合模式:
      - "concat"（默认）: 各自投影后拼接再投影
      - "gate": 门控融合 g * wav2vec + (1 - g) * librosa

    Input:  audio_wav2vec [B, N_word, 768], audio_librosa [B, N_word, 93]
    Output: [B, N_word, d_model]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        d_audio = cfg.d_audio
        self.fusion_mode = "gate" if cfg.use_audio_gate else cfg.audio_fusion_mode

        self.wav2vec_proj = nn.Linear(cfg.audio_wav2vec_dim, d_audio)
        self.librosa_proj = nn.Linear(cfg.audio_librosa_dim, d_audio)

        if self.fusion_mode == "concat":
            self.fuse_proj = nn.Linear(2 * d_audio, cfg.d_model)
        else:
            self.gate_fc = nn.Linear(2 * d_audio, d_audio)
            self.out_proj = nn.Linear(d_audio, cfg.d_model)

        self.mamba_enc = MambaEncoder(
            d_in=cfg.d_model,
            d_model=cfg.d_model,
            n_layers=cfg.mamba_n_layers,
            d_state=cfg.mamba_d_state,
            d_conv=cfg.mamba_d_conv,
            expand=cfg.mamba_expand,
        )

    def forward(
        self,
        wav2vec_feat: torch.Tensor,
        librosa_feat: torch.Tensor,
    ) -> torch.Tensor:
        # wav2vec_feat: [B, N_word, 768], librosa_feat: [B, N_word, 93]
        w = self.wav2vec_proj(wav2vec_feat)       # [B, N_word, d_audio]
        l = self.librosa_proj(librosa_feat)       # [B, N_word, d_audio]

        if self.fusion_mode == "concat":
            h = self.fuse_proj(torch.cat([w, l], dim=-1))  # [B, N_word, d_model]
        else:
            gate = torch.sigmoid(self.gate_fc(torch.cat([w, l], dim=-1)))
            h = self.out_proj(gate * w + (1 - gate) * l)   # [B, N_word, d_model]

        return self.mamba_enc(h)                  # [B, N_word, d_model]


# ============================================================================
#  (D) Text Encoder + Token-to-Word Pooling
# ============================================================================

def token_to_word_pool(
    text_repr: torch.Tensor,
    tok_timestamps: torch.Tensor,
    word_timestamps: torch.Tensor,
    tok_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """将 token 级表示通过时间戳对齐 mean-pool 到词级。

    对每个 word 时间窗 [w_start, w_end]，找到所有 tok_start ∈ [w_start, w_end] 的 token，
    取其 embedding 均值作为该 word 的表示。

    Args:
        text_repr:       [B, N_tok, D]
        tok_timestamps:  [B, N_tok, 2]  每个 token 的 (start, end)
        word_timestamps: [B, N_word, 2] 每个词的 (start, end)
        tok_mask:        [B, N_tok] True=有效 token

    Returns:
        [B, N_word, D]
    """
    B, N_tok, D = text_repr.shape
    N_word = word_timestamps.size(1)
    device = text_repr.device

    tok_start = tok_timestamps[:, :, 0]          # [B, N_tok]
    word_start = word_timestamps[:, :, 0]        # [B, N_word]
    word_end = word_timestamps[:, :, 1]          # [B, N_word]

    # [B, N_word, N_tok]: 判断每个 token 是否落入每个 word 的时间窗
    in_window = (
        (tok_start.unsqueeze(1) >= word_start.unsqueeze(2) - 1e-4)
        & (tok_start.unsqueeze(1) <= word_end.unsqueeze(2) + 1e-4)
    )  # [B, N_word, N_tok]

    if tok_mask is not None:
        in_window = in_window & tok_mask.unsqueeze(1)

    # 对 window 内 token 做加权平均（均匀权重）
    counts = in_window.float().sum(dim=2, keepdim=True).clamp(min=1)  # [B, N_word, 1]
    pooled = torch.bmm(in_window.float(), text_repr) / counts        # [B, N_word, D]
    return pooled


class TextEncoder(nn.Module):
    """文本编码器：BERT embedding → (可选 Mamba) → token-to-word pooling。

    当 text_use_mamba=True：Mamba 建模 token 依赖后对齐到 N_word。
    当 text_use_mamba=False：仅 Linear 投影 + token-to-word 池化，无 Mamba（实验用）。

    Input:  text_embedding [B, N_tok, D_text], text_token_timestamps, text_word_timestamps
    Output: [B, N_word, d_model]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        self.use_mamba = getattr(cfg, "text_use_mamba", True)
        if self.use_mamba:
            self.mamba_enc = MambaEncoder(
                d_in=cfg.text_dim,
                d_model=cfg.d_model,
                n_layers=cfg.mamba_n_layers,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
            )
            self.proj = None
        else:
            self.mamba_enc = None
            self.proj = nn.Linear(cfg.text_dim, cfg.d_model)

    def forward(
        self,
        text_emb: torch.Tensor,
        tok_timestamps: torch.Tensor,
        word_timestamps: torch.Tensor,
        tok_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # text_emb: [B, N_tok, D_text]
        if self.use_mamba:
            h = self.mamba_enc(text_emb)             # [B, N_tok, d_model]
        else:
            h = self.proj(text_emb)                 # [B, N_tok, d_model]
        h_word = token_to_word_pool(h, tok_timestamps, word_timestamps, tok_mask)
        return h_word                               # [B, N_word, d_model]

    def get_token_repr(self, text_emb: torch.Tensor) -> torch.Tensor:
        """返回 token 级表示 [B, N_tok, d_model]，供无 word 时间戳时 _align_to_word 使用。"""
        if self.use_mamba:
            return self.mamba_enc(text_emb)
        return self.proj(text_emb)


# ============================================================================
#  (E) Word-level Gated Cross-Modal Fusion (Stage 1)
# ============================================================================

class GatedCrossModalFusion(nn.Module):
    """词级门控跨模态融合。

    以文本为语义锚点 (Query)，分别向视觉和音频做 Cross-Attention 检索补充信息，
    再通过三路可学习门控逐词自适应调节各模态贡献权重。

    Input:  text_repr [B, N_word, D], visual_repr [B, N_word, D], audio_repr [B, N_word, D]
    Output: fused [B, N_word, D]
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        # Text ← Visual cross-attention
        self.cross_tv = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_tv = nn.LayerNorm(d_model)

        # Text ← Audio cross-attention
        self.cross_ta = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ta = nn.LayerNorm(d_model)

        # 三路门控
        self.gate_fc = nn.Linear(3 * d_model, 3 * d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        text_repr: torch.Tensor,
        visual_repr: torch.Tensor,
        audio_repr: torch.Tensor,
        word_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # text_repr, visual_repr, audio_repr: [B, N_word, D]
        # word_mask: [B, N_word] True=有效

        key_padding_mask = ~word_mask if word_mask is not None else None

        # Text ← Visual: text 做 Q，visual 做 K/V
        tv, _ = self.cross_tv(text_repr, visual_repr, visual_repr, key_padding_mask=key_padding_mask)
        tv = self.norm_tv(tv + text_repr)                       # [B, N_word, D]

        # Text ← Audio: text 做 Q，audio 做 K/V
        ta, _ = self.cross_ta(text_repr, audio_repr, audio_repr, key_padding_mask=key_padding_mask)
        ta = self.norm_ta(ta + text_repr)                       # [B, N_word, D]

        # 三路门控融合
        concat = torch.cat([tv, ta, text_repr], dim=-1)         # [B, N_word, 3D]
        gates = torch.sigmoid(self.gate_fc(concat))             # [B, N_word, 3D]
        g_v, g_a, g_t = gates.chunk(3, dim=-1)                 # 各 [B, N_word, D]
        fused = g_v * tv + g_a * ta + g_t * text_repr           # [B, N_word, D]
        return self.out_norm(fused)


# ============================================================================
#  (E') Simple Cross-Modal Attention（仅跨模态注意力，无门控/无 Stage2）
# ============================================================================


class SimpleCrossModalAttention(nn.Module):
    """单纯跨模态注意力：以某一模态为 Query，其余为 K/V 做 Cross-Attention，残差 + LayerNorm。

    由 use_visual / use_audio / use_text 控制参与融合的模态；优先以文本为 Query，
    若无文本则以视觉为 Query，再否则以音频为 Query。用于 ablation 与简化实验。

    Input:  visual_repr, audio_repr, text_repr 各 [B, N_word, D]（未参与模态可用零向量）
    Output: [B, N_word, D]
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        use_visual: bool = True,
        use_audio: bool = True,
        use_text: bool = True,
    ):
        super().__init__()
        self.use_visual = use_visual
        self.use_audio = use_audio
        self.use_text = use_text
        self._n_modals = sum([use_visual, use_audio, use_text])
        if self._n_modals == 0:
            raise ValueError("至少启用一种模态")

        # 最多 2 个 cross-attn：primary 对 other1、primary 对 other2
        self.cross_1 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_2 = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        visual_repr: torch.Tensor,
        audio_repr: torch.Tensor,
        text_repr: torch.Tensor,
        word_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # 按优先级收集：text, visual, audio
        parts: list[tuple[str, torch.Tensor]] = []
        if self.use_text:
            parts.append(("t", text_repr))
        if self.use_visual:
            parts.append(("v", visual_repr))
        if self.use_audio:
            parts.append(("a", audio_repr))

        if len(parts) == 1:
            return self.out_norm(parts[0][1])

        key_padding_mask = ~word_mask if word_mask is not None else None
        primary_name, primary_repr = parts[0]
        fused = primary_repr

        for i, (_, other_repr) in enumerate(parts[1:], start=1):
            cross = self.cross_1 if i == 1 else self.cross_2
            norm = self.norm_1 if i == 1 else self.norm_2
            out, _ = cross(primary_repr, other_repr, other_repr, key_padding_mask=key_padding_mask)
            fused = norm(fused + out)

        return self.out_norm(fused)


# ============================================================================
#  (F) Coupled Mamba Sequence-level Fusion (Stage 2)
# ============================================================================

class CoupledMambaLayer(nn.Module):
    """单层耦合 Mamba：三模态各自 Mamba 扫描 + 跨模态隐状态耦合注入。

    核心思想 (Coupled Mamba, NeurIPS 2024):
      h_v(t) = A_v·h_v(t-1) + B_v·x_v(t) + α · C_couple · h_a(t-1)
      h_a(t) = A_a·h_a(t-1) + B_a·x_a(t) + β · C_couple · h_t(t-1)
      h_t(t) = A_t·h_t(t-1) + B_t·x_t(t) + γ · C_couple · h_v(t-1)

    由于 mamba_ssm 封装了 SSM 内核，无法直接修改隐状态传递过程，
    这里采用"耦合残差注入"近似实现：
      x_v' = x_v + α · Linear(x_a)   — 在输入端注入跨模态信息
    从而让 Mamba 内部 SSM 在扫描时隐式融合跨模态信号。

    Input:  v, a, t 各 [B, N, D]
    Output: v', a', t' 各 [B, N, D]
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.mamba_v = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_a = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_t = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # 跨模态耦合投影
        self.couple_a2v = nn.Linear(d_model, d_model, bias=False)
        self.couple_t2a = nn.Linear(d_model, d_model, bias=False)
        self.couple_v2t = nn.Linear(d_model, d_model, bias=False)

        # 可学习耦合强度标量（初始化为较小值以稳定训练初期）
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.beta = nn.Parameter(torch.tensor(0.1))
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 耦合残差注入：在输入端混入跨模态信号
        v_in = v + self.alpha * self.couple_a2v(a)
        a_in = a + self.beta * self.couple_t2a(t)
        t_in = t + self.gamma * self.couple_v2t(v)

        v_out = self.mamba_v(v_in)
        a_out = self.mamba_a(a_in)
        t_out = self.mamba_t(t_in)
        return v_out, a_out, t_out


class CoupledMambaFusion(nn.Module):
    """多层耦合 Mamba 融合 + 拼接投影。

    Input:  visual_repr, audio_repr, text_repr 各 [B, N_word, D]
    Output: [B, N_word, D]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CoupledMambaLayer(
                cfg.d_model,
                d_state=cfg.mamba_d_state,
                d_conv=cfg.mamba_d_conv,
                expand=cfg.mamba_expand,
            )
            for _ in range(cfg.coupled_n_layers)
        ])
        self.out_proj = nn.Linear(3 * cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        v: torch.Tensor,
        a: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            v, a, t = layer(v, a, t)

        fused = torch.cat([v, a, t], dim=-1)    # [B, N_word, 3D]
        return self.norm(self.out_proj(fused))   # [B, N_word, D]


# ============================================================================
#  (H) Word-level Interaction Attention — 任务二
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """固定正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, n: int) -> torch.Tensor:
        return self.pe[:, :n, :]


def interleave_by_time(
    c_repr: torch.Tensor,
    p_repr: torch.Tensor,
    c_timestamps: torch.Tensor,
    p_timestamps: torch.Tensor,
    c_mask: torch.Tensor | None = None,
    p_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """按时间戳将咨询师/来访者词级表示交错拼接（batch 化）。

    Args:
        c_repr: [B, N_c, D]  咨询师词级表示
        p_repr: [B, N_p, D]  来访者词级表示
        c_timestamps: [B, N_c, 2]  咨询师词级时间戳 (start, end)
        p_timestamps: [B, N_p, 2]  来访者词级时间戳 (start, end)
        c_mask: [B, N_c]  True=有效
        p_mask: [B, N_p]  True=有效

    Returns:
        interleaved: [B, N_c+N_p, D]  时间排序后的交错序列
        role_ids:    [B, N_c+N_p]     0=咨询师, 1=来访者, -1=padding
        merged_mask: [B, N_c+N_p]     True=有效
    """
    B, N_c, D = c_repr.shape
    N_p = p_repr.size(1)
    N_total = N_c + N_p
    device = c_repr.device

    c_starts = c_timestamps[:, :, 0]  # [B, N_c]
    p_starts = p_timestamps[:, :, 0]  # [B, N_p]

    # 拼接后排序得到全局时间顺序的索引
    all_starts = torch.cat([c_starts, p_starts], dim=1)   # [B, N_total]
    all_reprs = torch.cat([c_repr, p_repr], dim=1)        # [B, N_total, D]

    role_raw = torch.cat([
        torch.zeros(B, N_c, device=device, dtype=torch.long),
        torch.ones(B, N_p, device=device, dtype=torch.long),
    ], dim=1)  # [B, N_total]

    if c_mask is not None and p_mask is not None:
        all_mask = torch.cat([c_mask, p_mask], dim=1)
    else:
        all_mask = torch.ones(B, N_total, dtype=torch.bool, device=device)

    # 将 padding 位置的时间设为极大值，使其排序到末尾
    sort_keys = all_starts.clone()
    sort_keys[~all_mask] = float("inf")
    sorted_indices = sort_keys.argsort(dim=1)  # [B, N_total]

    # 按排序索引重排
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(sorted_indices)
    interleaved = all_reprs[batch_idx, sorted_indices]       # [B, N_total, D]
    role_ids = role_raw[batch_idx, sorted_indices]            # [B, N_total]
    merged_mask = all_mask[batch_idx, sorted_indices]         # [B, N_total]

    role_ids[~merged_mask] = -1
    return interleaved, role_ids, merged_mask


class WordLevelInteractionAttention(nn.Module):
    """(H) 词级交互注意力模块。

    将咨询师和来访者的词级表示按时间交错，加入角色编码和位置编码，
    通过因果 Transformer 建模跨角色的时序响应模式。

    Input:
        counselor_repr: [B, N_c, D]   咨询师词级表示
        client_repr:    [B, N_p, D]   来访者词级表示
        c_timestamps:   [B, N_c, 2]   咨询师词级时间戳
        p_timestamps:   [B, N_p, 2]   来访者词级时间戳
    Output:
        therapist_interaction: [B, N_c, D]  经交互建模后的咨询师表示
        client_interaction:    [B, N_p, D]  经交互建模后的来访者表示
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.role_embedding = nn.Embedding(2, d)  # 0=counselor, 1=client
        self.pos_enc = SinusoidalPositionalEncoding(d)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=cfg.interaction_n_heads,
            dim_feedforward=4 * d,
            dropout=cfg.interaction_dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.interaction_n_layers)
        self.norm = nn.LayerNorm(d)

    def forward(
        self,
        counselor_repr: torch.Tensor,
        client_repr: torch.Tensor,
        c_timestamps: torch.Tensor,
        p_timestamps: torch.Tensor,
        c_mask: torch.Tensor | None = None,
        p_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, N_c, D = counselor_repr.shape
        N_p = client_repr.size(1)
        N_total = N_c + N_p

        interleaved, role_ids, merged_mask = interleave_by_time(
            counselor_repr, client_repr, c_timestamps, p_timestamps, c_mask, p_mask,
        )

        # 角色编码 + 位置编码
        role_emb = self.role_embedding(role_ids.clamp(min=0))  # [B, N_total, D]
        pos_emb = self.pos_enc(N_total).expand(B, -1, -1)     # [B, N_total, D]
        x = interleaved + role_emb + pos_emb

        # 因果 mask：每个位置只能关注自身和之前的位置
        causal_mask = nn.Transformer.generate_square_subsequent_mask(N_total, device=x.device)
        # padding mask: True → 被忽略
        src_key_padding_mask = ~merged_mask

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)  # [B, N_total, D]

        # 按角色分离：恢复原始顺序
        # 方法：构造反向索引将交错序列拆回各自位置
        c_out = torch.zeros_like(counselor_repr)
        p_out = torch.zeros_like(client_repr)
        c_count = torch.zeros(B, dtype=torch.long, device=x.device)
        p_count = torch.zeros(B, dtype=torch.long, device=x.device)

        for t in range(N_total):
            role_t = role_ids[:, t]               # [B]
            is_c = (role_t == 0)
            is_p = (role_t == 1)
            for b in range(B):
                if is_c[b] and c_count[b] < N_c:
                    c_out[b, c_count[b]] = x[b, t]
                    c_count[b] += 1
                elif is_p[b] and p_count[b] < N_p:
                    p_out[b, p_count[b]] = x[b, t]
                    p_count[b] += 1

        return c_out, p_out


# ============================================================================
#  (I) Multimodal Synchrony Tracking — 任务二
# ============================================================================

def temporal_resample(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """将变长序列 [B, N, D] 通过线性插值重采样到固定长度 target_len。"""
    # F.interpolate 需要 [B, D, N] 格式
    x_t = x.transpose(1, 2)                                   # [B, D, N]
    x_resampled = F.interpolate(x_t, size=target_len, mode="linear", align_corners=False)
    return x_resampled.transpose(1, 2)                         # [B, target_len, D]


class MultimodalSynchronyTracker(nn.Module):
    """(I) 多模态行为同步性追踪模块。

    将咨询师/来访者的词级多模态表示投影为行为状态向量，
    通过时间重采样对齐到统一时间轴后，在滑动窗口内计算：
      - 余弦相似度（整体方向一致性）
      - 皮尔逊相关（逐维度线性耦合）
      - L2 差异度（捕捉行为失谐）
    最终用 Mamba 编码同步性特征序列。

    Input:
        counselor_repr: [B, N_c, D]  咨询师词级表示
        client_repr:    [B, N_p, D]  来访者词级表示
    Output:
        synchrony_repr: [B, N_windows, d_model]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        self.n_align = cfg.sync_n_align
        self.window = cfg.sync_window
        self.stride = cfg.sync_stride

        self.behavior_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_sync),
            nn.GELU(),
            nn.Linear(cfg.d_sync, cfg.d_sync),
        )

        # 每窗口产生 3 个同步性标量: cos_sim, pearson, l2_diff
        # 总特征维度 = 3
        n_windows = max(1, (cfg.sync_n_align - cfg.sync_window) // cfg.sync_stride + 1)
        self.sync_mamba = MambaEncoder(
            d_in=3,
            d_model=cfg.d_model,
            n_layers=2,
            d_state=cfg.mamba_d_state,
            d_conv=min(cfg.mamba_d_conv, max(2, n_windows)),
            expand=cfg.mamba_expand,
        )

    def _windowed_sync(
        self,
        t_aligned: torch.Tensor,
        c_aligned: torch.Tensor,
    ) -> torch.Tensor:
        """滑动窗口同步性计算。

        Args:
            t_aligned: [B, N_align, d_sync]
            c_aligned: [B, N_align, d_sync]
        Returns:
            sync_feats: [B, N_windows, 3]  (cos_sim, pearson, l2_diff)
        """
        B = t_aligned.size(0)
        feats = []
        for start in range(0, self.n_align - self.window + 1, self.stride):
            tw = t_aligned[:, start : start + self.window, :]  # [B, W, d_sync]
            cw = c_aligned[:, start : start + self.window, :]

            # 展平窗口内所有维度为一个向量
            tw_flat = tw.reshape(B, -1)  # [B, W*d_sync]
            cw_flat = cw.reshape(B, -1)

            cos_sim = F.cosine_similarity(tw_flat, cw_flat, dim=-1)  # [B]

            # 皮尔逊相关 = 去均值后的余弦相似度
            tw_centered = tw_flat - tw_flat.mean(dim=-1, keepdim=True)
            cw_centered = cw_flat - cw_flat.mean(dim=-1, keepdim=True)
            tw_norm = tw_centered.norm(dim=-1).clamp(min=1e-8)
            cw_norm = cw_centered.norm(dim=-1).clamp(min=1e-8)
            pearson = (tw_centered * cw_centered).sum(dim=-1) / (tw_norm * cw_norm)

            l2_diff = (tw_flat - cw_flat).norm(dim=-1)  # [B]

            feats.append(torch.stack([cos_sim, pearson, l2_diff], dim=-1))  # [B, 3]

        return torch.stack(feats, dim=1)  # [B, N_windows, 3]

    def forward(
        self,
        counselor_repr: torch.Tensor,
        client_repr: torch.Tensor,
    ) -> torch.Tensor:
        t_behavior = self.behavior_proj(counselor_repr)  # [B, N_c, d_sync]
        c_behavior = self.behavior_proj(client_repr)     # [B, N_p, d_sync]

        t_aligned = temporal_resample(t_behavior, self.n_align)  # [B, N_align, d_sync]
        c_aligned = temporal_resample(c_behavior, self.n_align)

        sync_feats = self._windowed_sync(t_aligned, c_aligned)  # [B, N_windows, 3]
        return self.sync_mamba(sync_feats)                       # [B, N_windows, d_model]


# ============================================================================
#  (J) Alliance Dynamics Module — 任务二
# ============================================================================

class AllianceDynamicsModule(nn.Module):
    """(J) 治疗联盟动态评估模块。

    整合交互注意力、同步性追踪的输出，通过动态路由注意力融合
    五路特征（therapist_global, client_global, sync_global, diff, prod），
    输出最终 FIS 预测。

    Input:
        therapist_global: [B, D]  咨询师交互全局表示
        client_global:    [B, D]  来访者交互全局表示
        sync_global:      [B, D]  同步性全局表示
    Output:
        fis_scores: [B, n_labels]
    """

    def __init__(self, cfg: FISNetConfig):
        super().__init__()
        d = cfg.d_model
        n_routes = 5  # therapist, client, sync, diff, prod

        self.routing_fc = nn.Sequential(
            nn.Linear(n_routes * d, d),
            nn.GELU(),
            nn.Linear(d, n_routes),
        )

        self.head = nn.Sequential(
            nn.Linear(d, cfg.head_hidden),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(cfg.head_hidden, cfg.n_labels),
        )

    def forward(
        self,
        therapist_global: torch.Tensor,
        client_global: torch.Tensor,
        sync_global: torch.Tensor,
    ) -> torch.Tensor:
        diff_repr = therapist_global - client_global   # [B, D]
        prod_repr = therapist_global * client_global   # [B, D]

        inputs = [therapist_global, client_global, sync_global, diff_repr, prod_repr]
        concat_repr = torch.cat(inputs, dim=-1)        # [B, 5D]

        routing_weights = F.softmax(self.routing_fc(concat_repr), dim=-1)  # [B, 5]
        stacked = torch.stack(inputs, dim=1)            # [B, 5, D]
        routed = (routing_weights.unsqueeze(-1) * stacked).sum(dim=1)      # [B, D]

        return self.head(routed)                        # [B, n_labels]


# ============================================================================
#  FIS-Net 主模型
# ============================================================================

class FISNet(nn.Module):
    """Multimodal FIS Assessment Network — 支持任务一与任务二。

    融合模式由 fusion_mode 控制：
      - "simple_cross"（默认）: 去除 Stage1/Stage2，仅用跨模态注意力 + 参数调控模态
      - "stage1_stage2": 原 GatedCrossModalFusion + CoupledMambaFusion

    任务一前向流程 (task=1，仅咨询师) — fusion_mode="simple_cross" 时:
      1. VisualEncoder(video_openface3) → visual_repr [B, N_word, D]（use_visual 控制）
      2. AudioEncoder(wav2vec, librosa) → audio_repr  [B, N_word, D]（use_audio 控制）
      3. TextEncoder(text_emb) → text_repr [B, N_word, D]（use_text 控制；text_use_mamba=False 时无 Mamba）
      4. SimpleCrossModalAttention(visual, audio, text) → fused [B, N_word, D]（仅跨模态注意力）
      5. AttentivePooling → [B, D]
      6. MLP Head → [B, n_labels]

    任务一 — fusion_mode="stage1_stage2" 时:
      1-3. 同上
      4. Stage1: GatedCrossModalFusion → fused_s1
      5. Stage2: CoupledMambaFusion → fused_s2
      6-7. AttentivePooling → MLP Head

    任务二前向流程 (task=2，咨询师+来访者):
      1-4/5. 双塔编码：咨询师/来访者各自走上述步（参数独立或共享）
      6. (H) WordLevelInteractionAttention → therapist/client 交互表示
      7. (I) MultimodalSynchronyTracker  → synchrony 表示
      8. (J) AllianceDynamicsModule       → FIS 预测

    Ablation 开关（同时适用于两个任务）:
      - use_visual / use_audio / use_text: 参与融合的模态（simple_cross 时由 SimpleCrossModalAttention 使用）
      - fusion_mode: "simple_cross" | "stage1_stage2"
      - text_use_mamba: False 时文本无 Mamba，仅 Linear + token-to-word 池化
      - use_stage1_fusion / use_stage2_fusion: 仅 stage1_stage2 时生效
      - use_grouping_attn / use_interaction_attn / use_synchrony / use_alliance: 同上

    forward() 接口与 collate_fis_batch 输出对齐:
      task=1 batch:
        { "counselor": {...}, "counselor_word_mask", "counselor_tok_mask", "labels" }
      task=2 batch:
        上述 + { "patient": {...}, "patient_word_mask", "patient_tok_mask" }
    """

    def __init__(self, cfg: FISNetConfig | None = None, **kwargs: Any):
        super().__init__()
        if cfg is None:
            cfg = FISNetConfig(**kwargs)
        self.cfg = cfg

        # ---- 咨询师侧单模态编码器 ----
        if cfg.use_visual:
            self.visual_encoder = VisualEncoder(cfg)
        if cfg.use_audio:
            self.audio_encoder = AudioEncoder(cfg)
        if cfg.use_text:
            self.text_encoder = TextEncoder(cfg)

        # ---- 统计实际启用的模态数 ----
        self._n_modalities = sum([cfg.use_visual, cfg.use_audio, cfg.use_text])
        self._fusion_mode = getattr(cfg, "fusion_mode", "stage1_stage2")

        # ---- 融合：simple_cross 仅跨模态注意力；stage1_stage2 为原 Stage1 + Stage2 ----
        if self._fusion_mode == "simple_cross":
            self.stage1_fusion = None
            self.stage2_fusion = None
            self.fallback_proj = None
            if self._n_modalities >= 2:
                self.simple_cross_modal = SimpleCrossModalAttention(
                    d_model=cfg.d_model,
                    n_heads=cfg.cross_attn_heads,
                    dropout=cfg.cross_attn_dropout,
                    use_visual=cfg.use_visual,
                    use_audio=cfg.use_audio,
                    use_text=cfg.use_text,
                )
            else:
                self.simple_cross_modal = None
        else:
            self.simple_cross_modal = None
            # Stage 1: 词级跨模态融合
            if cfg.use_stage1_fusion and self._n_modalities >= 2 and cfg.use_text:
                self.stage1_fusion = GatedCrossModalFusion(
                    d_model=cfg.d_model,
                    n_heads=cfg.cross_attn_heads,
                    dropout=cfg.cross_attn_dropout,
                )
            else:
                self.stage1_fusion = None
            # Stage 2: Coupled Mamba 融合
            if cfg.use_stage2_fusion and self._n_modalities == 3:
                self.stage2_fusion = CoupledMambaFusion(cfg)
            else:
                self.stage2_fusion = None
            # 退化融合分支
            if self.stage2_fusion is None and self._n_modalities > 1:
                self.fallback_proj = nn.Linear(self._n_modalities * cfg.d_model, cfg.d_model)
            else:
                self.fallback_proj = None

        # ---- Attentive Pooling（任务一用，任务二中也作为子模块使用） ----
        self.pool = AttentivePooling(cfg.d_model, cfg.pool_hidden)

        # ---- 任务分支 ----
        if cfg.task == 2:
            self._build_task2_modules(cfg)
        else:
            # 任务一预测头
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, cfg.n_labels),
            )

    # ----------------------------------------------------------------
    #  任务二模块构建
    # ----------------------------------------------------------------

    def _build_task2_modules(self, cfg: FISNetConfig) -> None:
        """构建任务二专属模块：来访者侧编码器 + 交互/同步/联盟模块。"""

        # ---- 来访者侧编码器（共享或独立参数）----
        if cfg.share_encoders:
            self.patient_visual_encoder = self.visual_encoder if cfg.use_visual else None
            self.patient_audio_encoder = self.audio_encoder if cfg.use_audio else None
            self.patient_text_encoder = self.text_encoder if cfg.use_text else None
            self.patient_stage1_fusion = self.stage1_fusion
            self.patient_stage2_fusion = self.stage2_fusion
            self.patient_fallback_proj = self.fallback_proj
        else:
            if cfg.use_visual:
                self.patient_visual_encoder = VisualEncoder(cfg)
            else:
                self.patient_visual_encoder = None
            if cfg.use_audio:
                self.patient_audio_encoder = AudioEncoder(cfg)
            else:
                self.patient_audio_encoder = None
            if cfg.use_text:
                self.patient_text_encoder = TextEncoder(cfg)
            else:
                self.patient_text_encoder = None
            if self._fusion_mode == "simple_cross":
                self.patient_stage1_fusion = None
                self.patient_stage2_fusion = None
                self.patient_fallback_proj = None
            else:
                if cfg.use_stage1_fusion and self._n_modalities >= 2 and cfg.use_text:
                    self.patient_stage1_fusion = GatedCrossModalFusion(
                        d_model=cfg.d_model,
                        n_heads=cfg.cross_attn_heads,
                        dropout=cfg.cross_attn_dropout,
                    )
                else:
                    self.patient_stage1_fusion = None
                if cfg.use_stage2_fusion and self._n_modalities == 3:
                    self.patient_stage2_fusion = CoupledMambaFusion(cfg)
                else:
                    self.patient_stage2_fusion = None
                if self.patient_stage2_fusion is None and self._n_modalities > 1:
                    self.patient_fallback_proj = nn.Linear(self._n_modalities * cfg.d_model, cfg.d_model)
                else:
                    self.patient_fallback_proj = None

        # ---- (H) 词级交互注意力 ----
        if cfg.use_interaction_attn:
            self.interaction_attn = WordLevelInteractionAttention(cfg)
        else:
            self.interaction_attn = None

        # ---- (I) 多模态同步性追踪 ----
        if cfg.use_synchrony:
            self.synchrony_tracker = MultimodalSynchronyTracker(cfg)
        else:
            self.synchrony_tracker = None

        # ---- 来访者侧 Attentive Pooling ----
        self.pool_patient = AttentivePooling(cfg.d_model, cfg.pool_hidden)

        # ---- 同步性 Attentive Pooling ----
        if cfg.use_synchrony:
            self.pool_sync = AttentivePooling(cfg.d_model, cfg.pool_hidden)

        # ---- (J) 联盟动态模块 或 退化双塔 MLP ----
        if cfg.use_alliance and cfg.use_synchrony:
            self.alliance = AllianceDynamicsModule(cfg)
        else:
            # 退化方案：拼接可用全局表示 → MLP
            n_inputs = 2  # therapist + client
            if cfg.use_synchrony:
                n_inputs += 1
            self.alliance = None
            self.task2_head = nn.Sequential(
                nn.Linear(n_inputs * cfg.d_model, cfg.head_hidden),
                nn.GELU(),
                nn.Dropout(cfg.head_dropout),
                nn.Linear(cfg.head_hidden, cfg.n_labels),
            )
            self._task2_n_inputs = n_inputs

    # ----------------------------------------------------------------
    #  单角色编码管线（供双塔复用）
    # ----------------------------------------------------------------

    def _encode_role(
        self,
        role_data: dict[str, Any],
        word_mask: torch.Tensor,
        tok_mask: torch.Tensor | None,
        visual_enc: nn.Module | None,
        audio_enc: nn.Module | None,
        text_enc: nn.Module | None,
        stage1: nn.Module | None,
        stage2: nn.Module | None,
        fallback: nn.Module | None,
    ) -> torch.Tensor:
        """对单角色（咨询师或来访者）执行完整编码管线 → [B, N_word, D]。"""
        B = word_mask.size(0)
        N_word = word_mask.size(1)
        device = word_mask.device

        # 视觉编码
        if visual_enc is not None and "video_openface3" in role_data and role_data["video_openface3"] is not None:
            visual_repr = visual_enc(role_data["video_openface3"])
        else:
            visual_repr = self._zero_repr(B, N_word, device)

        # 音频编码
        if audio_enc is not None and "audio_wav2vec" in role_data and role_data["audio_wav2vec"] is not None:
            librosa_feat = role_data.get("audio_librosa")
            if librosa_feat is None:
                librosa_feat = torch.zeros(B, N_word, self.cfg.audio_librosa_dim, device=device)
            audio_repr = audio_enc(role_data["audio_wav2vec"], librosa_feat)
        else:
            audio_repr = self._zero_repr(B, N_word, device)

        # 文本编码
        if text_enc is not None and "text_embedding" in role_data and role_data["text_embedding"] is not None:
            tok_ts = role_data.get("text_token_timestamps")
            word_ts = role_data.get("text_word_timestamps")
            if tok_ts is not None and word_ts is not None:
                text_repr = text_enc(role_data["text_embedding"], tok_ts, word_ts, tok_mask)
            else:
                text_h = text_enc.get_token_repr(role_data["text_embedding"])
                text_repr = self._align_to_word(text_h, N_word)
        else:
            text_repr = self._zero_repr(B, N_word, device)

        # 简化融合：仅跨模态注意力（无 Stage1/Stage2）
        simple_cross = getattr(self, "simple_cross_modal", None)
        if simple_cross is not None:
            return simple_cross(visual_repr, audio_repr, text_repr, word_mask)

        # Stage 1
        if stage1 is not None:
            fused_s1 = stage1(text_repr, visual_repr, audio_repr, word_mask)
        else:
            fused_s1 = text_repr

        # Stage 2
        if stage2 is not None:
            fused = stage2(visual_repr, audio_repr, fused_s1)
        elif fallback is not None:
            parts = []
            if self.cfg.use_visual:
                parts.append(visual_repr)
            if self.cfg.use_audio:
                parts.append(audio_repr)
            if self.cfg.use_text:
                parts.append(fused_s1)
            fused = fallback(torch.cat(parts, dim=-1))
        else:
            if self.cfg.use_visual:
                fused = visual_repr
            elif self.cfg.use_audio:
                fused = audio_repr
            else:
                fused = fused_s1

        return fused  # [B, N_word, D]

    # ----------------------------------------------------------------
    #  辅助方法
    # ----------------------------------------------------------------

    def _zero_repr(self, B: int, N: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(B, N, self.cfg.d_model, device=device)

    def _align_to_word(self, token_repr: torch.Tensor, n_word: int) -> torch.Tensor:
        B, N_tok, D = token_repr.shape
        if N_tok >= n_word:
            return token_repr[:, :n_word, :]
        pad = torch.zeros(B, n_word - N_tok, D, device=token_repr.device)
        return torch.cat([token_repr, pad], dim=1)

    # ----------------------------------------------------------------
    #  Forward
    # ----------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            batch: collate_fis_batch 输出的 dict。

        Returns:
            [B, n_labels] FIS 预测分数。
        """
        if self.cfg.task == 2:
            return self._forward_task2(batch)
        return self._forward_task1(batch)

    def _forward_task1(self, batch: dict[str, Any]) -> torch.Tensor:
        """任务一：仅咨询师 → 编码 → Pooling → 预测。"""
        word_mask = batch["counselor_word_mask"]
        tok_mask = batch.get("counselor_tok_mask")

        fused = self._encode_role(
            batch["counselor"], word_mask, tok_mask,
            getattr(self, "visual_encoder", None),
            getattr(self, "audio_encoder", None),
            getattr(self, "text_encoder", None),
            self.stage1_fusion,
            self.stage2_fusion,
            self.fallback_proj,
        )

        pooled = self.pool(fused, word_mask)
        return self.head(pooled)

    def _forward_task2(self, batch: dict[str, Any]) -> torch.Tensor:
        """任务二：咨询师+来访者 → 双塔编码 → 交互/同步/联盟 → 预测。"""
        c_word_mask = batch["counselor_word_mask"]
        c_tok_mask = batch.get("counselor_tok_mask")
        p_word_mask = batch["patient_word_mask"]
        p_tok_mask = batch.get("patient_tok_mask")

        # ---- 1. 双塔编码 ----
        c_fused = self._encode_role(
            batch["counselor"], c_word_mask, c_tok_mask,
            getattr(self, "visual_encoder", None),
            getattr(self, "audio_encoder", None),
            getattr(self, "text_encoder", None),
            self.stage1_fusion,
            self.stage2_fusion,
            self.fallback_proj,
        )  # [B, N_c, D]

        p_fused = self._encode_role(
            batch["patient"], p_word_mask, p_tok_mask,
            getattr(self, "patient_visual_encoder", None),
            getattr(self, "patient_audio_encoder", None),
            getattr(self, "patient_text_encoder", None),
            getattr(self, "patient_stage1_fusion", None),
            getattr(self, "patient_stage2_fusion", None),
            getattr(self, "patient_fallback_proj", None),
        )  # [B, N_p, D]

        # ---- 2. (H) 词级交互注意力 ----
        if self.interaction_attn is not None:
            c_ts = batch["counselor"].get("text_word_timestamps")
            p_ts = batch["patient"].get("text_word_timestamps")
            if c_ts is not None and p_ts is not None:
                c_interact, p_interact = self.interaction_attn(
                    c_fused, p_fused, c_ts, p_ts, c_word_mask, p_word_mask,
                )
            else:
                c_interact, p_interact = c_fused, p_fused
        else:
            c_interact, p_interact = c_fused, p_fused

        # ---- 3. 全局池化 ----
        c_global = self.pool(c_interact, c_word_mask)               # [B, D]
        p_global = self.pool_patient(p_interact, p_word_mask)       # [B, D]

        # ---- 4. (I) 同步性追踪 ----
        if self.synchrony_tracker is not None:
            sync_repr = self.synchrony_tracker(c_fused, p_fused)    # [B, N_windows, D]
            sync_global = self.pool_sync(sync_repr)                 # [B, D]
        else:
            sync_global = torch.zeros_like(c_global)

        # ---- 5. (J) 联盟动态 → 预测 ----
        if self.alliance is not None:
            return self.alliance(c_global, p_global, sync_global)
        else:
            parts = [c_global, p_global]
            if self.cfg.use_synchrony:
                parts.append(sync_global)
            return self.task2_head(torch.cat(parts, dim=-1))


# ============================================================================
#  测试入口
# ============================================================================

def _make_role_feats(
    batch_size: int,
    n_word: int,
    n_tok: int,
    video_dim: int,
    text_dim: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """构造单角色的假特征 dict + word_mask + tok_mask。"""
    word_mask = torch.ones(batch_size, n_word, dtype=torch.bool, device=device)
    word_mask[:, n_word - 4:] = False

    tok_mask = torch.ones(batch_size, n_tok, dtype=torch.bool, device=device)
    tok_mask[:, n_tok - 8:] = False

    word_starts = torch.sort(torch.rand(batch_size, n_word, device=device), dim=1).values
    word_ts = torch.stack([word_starts, word_starts + 0.1], dim=-1)
    tok_starts = torch.sort(torch.rand(batch_size, n_tok, device=device), dim=1).values
    tok_ts = torch.stack([tok_starts, tok_starts + 0.05], dim=-1)

    feats = {
        "video_openface3": torch.randn(batch_size, n_word, video_dim, device=device),
        "audio_wav2vec": torch.randn(batch_size, n_word, 768, device=device),
        "audio_librosa": torch.randn(batch_size, n_word, 93, device=device),
        "text_embedding": torch.randn(batch_size, n_tok, text_dim, device=device),
        "text_word_timestamps": word_ts,
        "text_token_timestamps": tok_ts,
    }
    return feats, word_mask, tok_mask


def _make_fake_batch(
    batch_size: int = 2,
    n_word: int = 32,
    n_tok: int = 64,
    video_dim: int = 235,
    text_dim: int = 1024,
    task: int = 1,
    n_word_patient: int = 24,
    n_tok_patient: int = 48,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """构造与 collate_fis_batch 格式一致的假 batch。

    task=1: 仅含 counselor 数据
    task=2: 含 counselor + patient 数据
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    c_feats, c_word_mask, c_tok_mask = _make_role_feats(
        batch_size, n_word, n_tok, video_dim, text_dim, device,
    )

    batch: dict[str, Any] = {
        "counselor": c_feats,
        "counselor_word_mask": c_word_mask,
        "counselor_tok_mask": c_tok_mask,
        "labels": torch.randn(batch_size, 9, device=device),
    }

    if task == 2:
        p_feats, p_word_mask, p_tok_mask = _make_role_feats(
            batch_size, n_word_patient, n_tok_patient, video_dim, text_dim, device,
        )
        batch["patient"] = p_feats
        batch["patient_word_mask"] = p_word_mask
        batch["patient_tok_mask"] = p_tok_mask

    return batch


def _print_batch_shapes(batch: dict[str, Any], role: str = "counselor") -> None:
    data = batch[role]
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {role}.{k}: {tuple(v.shape)}")
    print(f"  {role}_word_mask: {tuple(batch[f'{role}_word_mask'].shape)}")
    print(f"  {role}_tok_mask:  {tuple(batch[f'{role}_tok_mask'].shape)}")


def _test_one_task(task: int, device: torch.device) -> None:
    """对指定任务执行 eval forward + train forward+backward 测试。"""
    print(f"\n{'='*50}")
    print(f" 测试 task={task}")
    print(f"{'='*50}")

    cfg = FISNetConfig(
        task=task,
        d_model=64,
        n_labels=9,
        video_dim=235,
        mamba_n_layers=2,
        coupled_n_layers=1,
        interaction_n_layers=1,
        pool_hidden=32,
        head_hidden=64,
        d_sync=32,
        sync_n_align=32,
        sync_window=8,
        sync_stride=4,
    )
    model = FISNet(cfg).to(device)
    batch = _make_fake_batch(batch_size=2, n_word=32, n_tok=64, task=task, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {n_params:,}")

    print("\n--- 输入形状 ---")
    _print_batch_shapes(batch, "counselor")
    if task == 2:
        _print_batch_shapes(batch, "patient")
    print(f"  labels: {tuple(batch['labels'].shape)}")

    model.eval()
    with torch.no_grad():
        out = model(batch)
    print(f"\n--- 输出形状 ---")
    print(f"  logits: {tuple(out.shape)}  # [B, n_labels]")

    model.train()
    pred = model(batch)
    loss = F.mse_loss(pred, batch["labels"])
    loss.backward()
    print(f"\n--- 梯度检查 ---")
    print(f"  loss: {loss.item():.6f}, backward 完成")
    print(f"\n--- task={task} 测试通过 ---")


def main() -> None:
    """测试 FISNet 任务一和任务二的输入输出。

    运行: python -m experiment.model.FisNet
    依赖: mamba_ssm（pip install mamba-ssm causal-conv1d）。
    """
    import sys

    try:
        from mamba_ssm import Mamba  # noqa: F401
    except ImportError:
        print("请先安装 mamba_ssm: pip install mamba-ssm causal-conv1d", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    _test_one_task(task=1, device=device)
    _test_one_task(task=2, device=device)

    print(f"\n{'='*50}")
    print(" 全部测试通过")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
