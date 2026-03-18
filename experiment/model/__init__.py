"""模型构建入口。"""

from __future__ import annotations

from typing import Any

from .bilstm_attn import BiLSTMAttnConfig, BiLSTMAttnFIS
from .DLinear import DLinearFIS, DLinearFISConfig
from .fis_regressor import FISSimpleMultimodalRegressor, FISTextRegressor
from .KANAD import KANADFIS, KANADFISConfig
from .MSGNet import MSGNetFIS, MSGNetFISConfig
from .Reformer import ReformerFIS, ReformerFISConfig
from .SegRNN import SegRNNFIS, SegRNNFISConfig
from .TCDyFIS import TCDyFIS, TCDyFISConfig
from .TCDyFIS_v2 import TCDyFISv2, TCDyFISv2Config
from .timefilter_fis import TimeFilterFIS, TimeFilterFISConfig
from .timesformer_fis import TimeSformerFIS, TimeSformerFISConfig


def build_model(name: str, kwargs: dict[str, Any]):
    name = name.lower()
    if name == "text_regressor":
        return FISTextRegressor(**kwargs)
    if name == "simple_multimodal":
        return FISSimpleMultimodalRegressor(**kwargs)
    if name in ("fis_net", "fisnet"):
        from .FisNet import FISNet, FISNetConfig
        cfg = FISNetConfig(**kwargs)
        return FISNet(cfg)
    if name in ("tcdyfis", "tc_dy_fis"):
        cfg = TCDyFISConfig(**kwargs)
        return TCDyFIS(cfg)
    if name in ("tcdyfis_v2", "tcdyfisv2"):
        cfg = TCDyFISv2Config(**kwargs)
        return TCDyFISv2(cfg)
    if name in ("bilstm_attn", "bilstm"):
        cfg = BiLSTMAttnConfig(**kwargs)
        return BiLSTMAttnFIS(cfg)
    if name in ("dlinear",):
        cfg = DLinearFISConfig(**kwargs)
        return DLinearFIS(cfg)
    if name in ("segrnn", "seg_rnn"):
        cfg = SegRNNFISConfig(**kwargs)
        return SegRNNFIS(cfg)
    if name in ("kanad",):
        cfg = KANADFISConfig(**kwargs)
        return KANADFIS(cfg)
    if name in ("reformer",):
        cfg = ReformerFISConfig(**kwargs)
        return ReformerFIS(cfg)
    if name in ("msgnet",):
        cfg = MSGNetFISConfig(**kwargs)
        return MSGNetFIS(cfg)
    if name in ("timefilter_fis", "timefilter"):
        cfg = TimeFilterFISConfig(**kwargs)
        return TimeFilterFIS(cfg)
    if name in ("timesformer_fis", "timesformer"):
        cfg = TimeSformerFISConfig(**kwargs)
        return TimeSformerFIS(cfg)
    raise ValueError(f"未知模型: {name}")

