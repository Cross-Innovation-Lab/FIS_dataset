#!/usr/bin/env python3
"""
绘制 FIS 各维度标签得分的分布图，用于论文。
- 柱状图：8 个指标均分（不含 All Ratings）
- 小提琴图：标出 25%/50%/75% 分位，箱线置于小提琴一侧并排
- MM 期刊偏好配色（色盲友好、印刷清晰）
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors as mcolors


def _lighten_hex(hex_color: str, factor: float = 0.75) -> str:
    """将颜色与白色混合得到浅色，factor 越大越接近原色。"""
    rgb = np.array(mcolors.to_rgb(hex_color))
    white = np.array([1.0, 1.0, 1.0])
    light = rgb * factor + white * (1 - factor)
    return mcolors.to_hex(np.clip(light, 0, 1))

# 论文用图：字体与线宽
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["figure.dpi"] = 150

# PDF 输出尺寸（英寸）：适配期刊单/双栏，略大以便清晰
FIG_BAR = (7.5, 3.5)       # 柱状图
FIG_GROUP = (6.0, 3.8)     # 分组小提琴
FIG_ALL = (9, 4.5)       # 八指标总图
# 较高 dpi 保证清晰度
DPI_PDF = 300
DPI_PNG = 300

# 标签列（不含 All Ratings，顺序固定便于分组）
LABEL_COLS = [
    "abc",
    "arrr",
    "emotional_expression",
    "empathy",
    "hope_and_pe",
    "persuasiveness",
    "verbal_fluency",
    "wau",
]
# Emotional Expression, Empathy, Hope and Positive Expectations, Persuasiveness, Verbal Fluency, Alliance Rupture-Repair Responsiveness, Alliance Bond Capacity, and Warmth, Acceptance, and Understanding
# 论文中显示的维度名称
LABEL_DISPLAY = {
    "abc": "Alliance Bond Capacity",
    "all_ratings": "All Ratings",
    "arrr": "Alliance Rupture-Repair Responsiveness",
    "emotional_expression": "Emotional Expression",
    "empathy": "Empathy",
    "hope_and_pe": "Hope and Positive Expectations",
    "persuasiveness": "Persuasiveness",
    "verbal_fluency": "Verbal Fluency",
    "wau": "Warmth, Acceptance, and Understanding",
}

# 每 3/3/2 个维度一组，共 3 组（8 个指标不含 All Ratings）
DIMENSION_GROUPS: list[list[str]] = [
    ["abc", "arrr", "emotional_expression"],
    ["empathy", "hope_and_pe", "persuasiveness"],
    ["verbal_fluency", "wau"],
]

# MM/ACM 期刊常用配色：色盲友好、印刷对比度好（基于 Okabe–Ito / Tableau 风格）
PALETTE_MM = ["#0173B2", "#DE8F05", "#029E73"]  # 蓝、橙、绿

# 九指标总图：柔和、不过艳的配色（低饱和度蓝绿灰系）
def _muted_palette(n: int) -> list[str]:
    """生成 n 个柔和色（灰蓝绿系，避免过于鲜艳）。"""
    cmap = plt.cm.GnBu(np.linspace(0.35, 0.82, n))
    return [mcolors.to_hex(c) for c in cmap]


PALETTE_ALL_MUTED = _muted_palette(8)

# 横轴标签：过长则换行，最大字符数（约）
MAX_CHARS_PER_LINE = 18


def _wrap_label(text: str, max_chars: int = MAX_CHARS_PER_LINE) -> str:
    """过长标签按空格换行，尽量保持词语完整。"""
    if len(text) <= max_chars:
        return text
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for w in words:
        need = len(w) + (1 if current else 0)  # 空格
        if current and sum(len(x) for x in current) + len(current) - 1 + need > max_chars:
            lines.append(" ".join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _wrap_labels(labels: list[str], max_chars: int = MAX_CHARS_PER_LINE) -> list[str]:
    """对标签列表逐条换行。"""
    return [_wrap_label(s, max_chars) for s in labels]


def load_and_melt(csv_path: Path, label_cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=label_cols)
    return df.melt(
        id_vars=["ID", "folder"] if "folder" in df.columns else ["ID"],
        value_vars=label_cols,
        var_name="dimension",
        value_name="score",
    )


def plot_mean_bar(wide_df: pd.DataFrame, out_path: Path) -> None:
    """绘制 8 个指标均分的柱状图（不含 All Ratings）。"""
    means = wide_df[LABEL_COLS].mean()
    order = [LABEL_DISPLAY[c] for c in LABEL_COLS]
    x = np.arange(len(order))
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(order)))

    fig, ax = plt.subplots(figsize=FIG_BAR)
    bars = ax.bar(x, means.reindex(LABEL_COLS).values, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_wrap_labels(order), rotation=20, ha="right")
    ax.set_ylabel("Mean Score", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylim(0, 5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    dpi = DPI_PDF if out_path.suffix == ".pdf" else DPI_PNG
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def _add_quartile_lines(ax, positions: list[float], widths: list[float], quartiles: list[list[float]], colors: list[str]) -> None:
    """在小提琴内绘制 25%/50%/75% 分位线。quartiles[i] = [q25, q50, q75]。"""
    for i, (pos, w, qs) in enumerate(zip(positions, widths, quartiles)):
        for q in qs:
            ax.hlines(q, pos - w / 2, pos + w / 2, colors=colors[i], linewidths=1.2, linestyles="solid")
        # 中位线加粗
        ax.hlines(qs[1], pos - w / 2, pos + w / 2, colors=colors[i], linewidths=2.0, linestyles="solid")


def plot_group_violin_box(
    long_df: pd.DataFrame,
    dimensions: list[str],
    out_path: Path,
    palette: list[str],
    *,
    y_range: tuple[float, float] = (1.0, 5.0),
    figsize: tuple[float, float] = FIG_GROUP,
) -> None:
    """绘制一组（3 个）维度：小提琴（内含 25/50/75% 分位线）+ 箱线置于一侧并排。"""
    df = long_df[long_df["dimension"].isin(dimensions)].copy()
    order = [LABEL_DISPLAY[d] for d in dimensions]
    # 每维度数据与分位数
    datas = [df[df["dimension"] == d]["score"].values for d in dimensions]
    quartiles = [np.percentile(d, [25, 50, 75]).tolist() for d in datas]

    fig, ax = plt.subplots(figsize=figsize)
    n = len(dimensions)
    width_violin = 0.38
    width_box = 0.22
    # 组间距 <1 以拉近各对象
    spacing = 0.68
    box_offset = width_violin / 2 + width_box / 2 + 0.02
    pos_violin = np.arange(n) * spacing
    pos_box = pos_violin + box_offset

    # 1. 小提琴
    vp = ax.violinplot(
        datas,
        positions=pos_violin,
        widths=width_violin,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.85)
        pc.set_edgecolor(palette[i])
        pc.set_linewidth(1.0)
    # 2. 小提琴内 25/50/75% 分位线
    _add_quartile_lines(ax, pos_violin.tolist(), [width_violin] * n, quartiles, palette)

    # 3. 箱线（置于小提琴右侧，与对应小提琴同色系浅色）
    bp = ax.boxplot(
        datas,
        positions=pos_box.tolist(),
        widths=width_box,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(facecolor="white", edgecolor="gray", linewidth=1.0),
        whiskerprops=dict(color="gray", linewidth=1.0),
        capprops=dict(color="gray", linewidth=1.0),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_lighten_hex(palette[i], factor=0.82))
        box.set_edgecolor(palette[i])
        box.set_linewidth(1.0)
    # 在 50% 中位线旁标注分数
    for i in range(n):
        med = quartiles[i][1]
        ax.text(
            pos_box[i] + width_box / 2 + 0.03,
            med,
            f"{med:.2f}",
            fontsize=8,
            ha="left",
            va="center",
            color="black",
        )

    ax.set_ylim(y_range)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticks((pos_violin + pos_box) / 2)
    ax.set_xticklabels(_wrap_labels(order), rotation=12, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    dpi = DPI_PDF if out_path.suffix == ".pdf" else DPI_PNG
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def plot_all_nine_violin_box(
    long_df: pd.DataFrame,
    out_path: Path,
    palette: list[str],
    *,
    y_range: tuple[float, float] = (1.0, 5.0),
    figsize: tuple[float, float] = FIG_ALL,
) -> None:
    """绘制全部 8 个维度一张图：小提琴（25/50/75% 分位线）+ 箱线置于小提琴右侧，柔和配色（不含 All Ratings）。"""
    dimensions = LABEL_COLS
    order = [LABEL_DISPLAY[d] for d in dimensions]
    datas = [long_df[long_df["dimension"] == d]["score"].values for d in dimensions]
    quartiles = [np.percentile(d, [25, 50, 75]).tolist() for d in datas]

    fig, ax = plt.subplots(figsize=figsize)
    n = len(dimensions)
    width_violin = 0.32
    width_box = 0.14
    spacing = 0.65  # 拉近 8 个对象间距
    box_offset = width_violin / 2 + width_box / 2 + 0.02
    pos_violin = np.arange(n) * spacing
    pos_box = pos_violin + box_offset

    vp = ax.violinplot(
        datas,
        positions=pos_violin,
        widths=width_violin,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for i, pc in enumerate(vp["bodies"]):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.88)
        pc.set_edgecolor(palette[i])
        pc.set_linewidth(0.9)
    _add_quartile_lines(ax, pos_violin.tolist(), [width_violin] * n, quartiles, palette)

    bp = ax.boxplot(
        datas,
        positions=pos_box.tolist(),
        widths=width_box,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.2),
        boxprops=dict(facecolor="white", edgecolor="gray", linewidth=0.9),
        whiskerprops=dict(color="gray", linewidth=0.9),
        capprops=dict(color="gray", linewidth=0.9),
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_lighten_hex(palette[i], factor=0.82))
        box.set_edgecolor(palette[i])
        box.set_linewidth(0.9)
    for i in range(n):
        med = quartiles[i][1]
        ax.text(
            pos_box[i] + width_box / 2 + 0.02,
            med,
            f"{med:.2f}",
            fontsize=7,
            ha="left",
            va="center",
            color="black",
        )

    ax.set_ylim(y_range)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_xlabel("")
    ax.set_xticks((pos_violin + pos_box) / 2)
    ax.set_xticklabels(_wrap_labels(order), rotation=20, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    dpi = DPI_PDF if out_path.suffix == ".pdf" else DPI_PNG
    plt.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    csv_path = base / "dataset" / "all_labels_Valid.csv"
    out_dir = base / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_df = pd.read_csv(csv_path).dropna(subset=LABEL_COLS)
    long_df = load_and_melt(csv_path, LABEL_COLS)

    # 1. 九指标均分柱状图
    plot_mean_bar(wide_df, out_dir / "fis_mean_scores_bar.pdf")
    plot_mean_bar(wide_df, out_dir / "fis_mean_scores_bar.png")

    # 2. 九指标总图：小提琴+分位线+箱线在右侧，柔和配色
    plot_all_nine_violin_box(long_df, out_dir / "fis_label_distribution_all.pdf", palette=PALETTE_ALL_MUTED)
    plot_all_nine_violin_box(long_df, out_dir / "fis_label_distribution_all.png", palette=PALETTE_ALL_MUTED)

    # 3. 分组小提琴+箱线（分位线 25/50/75%，箱线置于小提琴右侧）
    for i, dims in enumerate(DIMENSION_GROUPS, start=1):
        stem = f"fis_label_distribution_group{i}"
        palette = PALETTE_MM if len(dims) == 3 else PALETTE_MM[:2]  # 第 3 组仅 2 个维度
        plot_group_violin_box(
            long_df,
            dims,
            out_dir / f"{stem}.pdf",
            palette=palette,
        )
        plot_group_violin_box(
            long_df,
            dims,
            out_dir / f"{stem}.png",
            palette=palette,
        )

    print(f"Figures saved under {out_dir}")
    print("  Bar: fis_mean_scores_bar.pdf/png (8 dimensions mean, no All Ratings)")
    print("  All: fis_label_distribution_all.pdf/png (8 dims, violin+box right, muted colors)")
    print("  Group 1–3: fis_label_distribution_group1/2/3")


if __name__ == "__main__":
    main()
