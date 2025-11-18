#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# ===== Global style settings =====
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.unicode_minus"] = False

# ===== Generic pie function with legend =====
def make_donut_pie(ax, labels, counts, title, n_total=None):
    counts = np.array(counts, dtype=float)
    total = counts.sum()
    if n_total is None:
        n_total = int(total)

    # 只在扇形里显示百分比，避免太挤
    def autopct(pct):
        if pct < 1.0:      # 很小的比例直接不显示
            return ""
        return f"{pct:.1f}%"

    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(labels)))

    # 不在饼图上写标签，标签统一交给 legend
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=None,
        autopct=autopct,
        startangle=90,
        counterclock=False,
        colors=colors,
        pctdistance=0.7,
        wedgeprops=dict(
            width=1.0,      # 实心饼图
            edgecolor="white",
            linewidth=1.0
        ),
        textprops=dict(color="black")
    )

    # 图例内容：label + count + percent
    legend_labels = [
        f"{lab}: {int(c)} cases ({c / total * 100:.1f}%)"
        for lab, c in zip(labels, counts)
    ]
    ax.legend(
        wedges,
        legend_labels,
        loc="upper left",          # 图例以左上角为锚点
        bbox_to_anchor=(1.02, 1.0),# 锚点放在坐标(1.02, 1.0): 子图右上角稍微偏右
        fontsize=8,
        frameon=False
    )

    for t in autotexts:
        t.set_fontsize(8)

    ax.set_title(f"{title}", fontsize=12, pad=8)
    ax.set_aspect("equal")


# ===== Data (60 HCC cases) =====
N_TOTAL = 60

# 1. AJCC-TNM
ajcc_labels = [
    "IA",
    "IB",
    "II",
    "IIIA",
    "IIIB",
    "IVA",
    "IVB",
]

ajcc_counts = [12, 13, 10, 10, 9, 2, 4]

# 2. CNLC
cnlc_labels = ["Ia", "Ib", "IIa", "IIIa", "IIIb"]
cnlc_counts = [19, 13, 2, 17, 9]

# 3. BCLC
bclc_labels = ["0", "A", "B", "C", "D"]
bclc_counts = [7, 25, 4, 23, 1]

# 4. ECOG performance status (0 and 1)
ecog_labels = ["PS 0", "PS 1"]
ecog_counts = [31 + 25, 4]  # 56 with PS 0, 4 with PS 1

# 5. Child-Pugh class
child_labels = ["Class A", "Class B"]
child_counts = [53, 7]

# 6. Age bins
age_labels = ["60–69", "50–59", "70–79", "40–49", "<40", "≥80"]
age_counts = [25, 16, 12, 3, 2, 2]

# ===== Plot: 2 x 3 subplots =====
fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 8),          # 拉宽一点给图例留位置
    subplot_kw=dict(aspect="equal")
)

# Row 1
make_donut_pie(axes[0, 0], ajcc_labels, ajcc_counts, "AJCC-TNM stage", n_total=N_TOTAL)
make_donut_pie(axes[0, 1], cnlc_labels, cnlc_counts, "CNLC stage", n_total=N_TOTAL)
make_donut_pie(axes[0, 2], bclc_labels, bclc_counts, "BCLC stage", n_total=N_TOTAL)

# Row 2
make_donut_pie(axes[1, 0], ecog_labels, ecog_counts, "ECOG performance status", n_total=N_TOTAL)
make_donut_pie(axes[1, 1], child_labels, child_counts, "Child-Pugh class", n_total=N_TOTAL)
make_donut_pie(axes[1, 2], age_labels, age_counts, "Age distribution", n_total=N_TOTAL)

plt.tight_layout()
plt.savefig('60_stats.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('60_stats.png', dpi=300, bbox_inches='tight')
plt.show()
