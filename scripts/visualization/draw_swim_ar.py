import json
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from typing import List

def color_for_index(palette, i):
    """假设已有的辅助函数，用于根据索引返回调色板中的颜色。"""
    return palette[i % len(palette)]

def plot_total_reason_box(df: pd.DataFrame, out_svg: str, out_png: str, palette: List[str]):
    # fig C：仅剔除 TOTAL_ABC 三模型相同的病例
    models = ["Ours","GPT-4o","DeepSeek-R1"]

    data = [df.loc[(df["model"] == m), "TOTAL_ABC"].dropna().values for m in models]

    colors = [color_for_index(palette, i) for i in range(len(models))]
    plt.figure(figsize=(8,6))
    bp = plt.boxplot(data, labels=models, showmeans=False, patch_artist=True, showfliers=False)

    # 着色
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)
    for whisk in bp['whiskers']:
        whisk.set_color("#333333"); whisk.set_linewidth(1.0)
    for cap in bp['caps']:
        cap.set_color("#333333"); cap.set_linewidth(1.0)
    for median in bp['medians']:
        median.set_color("#111111"); median.set_linewidth(1.2)
    if 'means' in bp:
        for mean in bp['means']:
            mean.set_marker('-'); mean.set_markerfacecolor("#ffffff")
            mean.set_markeredgecolor("#111111"); mean.set_markersize(5)

    plt.ylabel("Total Score",fontsize=14)
    plt.tight_layout()
    plt.savefig(out_svg); plt.savefig(out_png, dpi=300); plt.close()

def plot_reason_parts_with_ci(df, out_svg: str, out_png: str, palette: list):
    """
    仅 A/B/C 的映射后均值（逐分项剔除相同病例），并在柱状图中加上 95% CI (置信区间)。
    """

    parts = ["evidence_completeness", "evidence_relevance", "guideline_citation_accuracy"]
    labels = ["Completeness", "Correctness", "Consistency"]
    models = ["Ours", "GPT-4o", "DeepSeek-R1"]

    # 存放每个模型对应各项的均值和误差（95% CI）
    means = {}
    ci_95 = {}

    for m in models:
        part_means = []
        part_ci = []
        for p in parts:
            series = pd.to_numeric(df.loc[df["model"] == m, p], errors="coerce").dropna()
            n = len(series)
            if n > 1:
                mean_val = np.mean(series)
                std_val = np.std(series, ddof=1)
                # 这里使用 1.96 * SE (SE = std / sqrt(n)) 近似计算 95% CI
                se = std_val / np.sqrt(n)
                margin = 1.96 * se
            elif n == 1:
                # 如果只有一个样本，SE = 0，这样置信区间为 0
                mean_val = float(series.iloc[0])
                margin = 0.0
            else:
                # 没有数据时，给个空值（或 nan）
                mean_val = np.nan
                margin = 0.0

            part_means.append(mean_val)
            part_ci.append(margin)
        means[m] = part_means
        ci_95[m] = part_ci

    # 开始绘图
    x = np.arange(len(parts)) * 0.8
    width = 0.20

    plt.figure(figsize=(10, 6))

    for i, m in enumerate(models):
        color = color_for_index(palette, i)
        # 这里用 yerr=ci_95[m] 传入误差，error_kw 可以指定误差线样式
        plt.bar(
            x + (i - 1) * width,
            means[m],
            yerr=ci_95[m],
            width=width,
            label=m,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            error_kw=dict(ecolor='black', capsize=4, lw=1)
        )

    plt.xticks(x, labels, rotation=0, ha="center",fontsize=14)
    plt.ylabel("Score (mean) ± 95% CI",fontsize=14)
    plt.legend(frameon=False)
    plt.tight_layout()

    plt.savefig(out_svg)
    plt.savefig(out_png, dpi=300)
    plt.close()

# 1. 读取 JSON 文件并整理为 DataFrame
json_file = "xxx"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

mapping = {
    'doctor_0': 'Ours',
    'doctor_1': 'GPT-4o',
    'doctor_2': 'DeepSeek-R1'}

rows = []
op_idx = 0
pbar_scot = []
for patient_id, metrics_dict in data.items():
    score_by_model = {
        'Ours': {'model': 'Ours'},
        'GPT-4o': {'model': 'GPT-4o'},
        'DeepSeek-R1': {'model': 'DeepSeek-R1'},
    }
    for dimension, doctor_scores in metrics_dict.items():
        for doctor, score in doctor_scores.items():
            # 这里添加一个随机数用来当作散点大小，模拟“大小略微不同”
            point_size = np.random.uniform(1, 20)
            rows.append({
                "patient_id": patient_id,
                "dimension": dimension,  # "evidence_completeness" 等
                "model": mapping[doctor],        # "doctor_0" / "doctor_1" / "doctor_2"
                "score": score,
                "point_size": point_size,
            })
            score_by_model[mapping[doctor]][dimension] = score
    for model_key, score_dict in score_by_model.items():
        score_dict["TOTAL_ABC"] = sum([
            score_dict.get("evidence_completeness", 0),
            score_dict.get("evidence_relevance", 0),
            score_dict.get("guideline_citation_accuracy", 0),
        ])
        pbar_scot.append(score_dict)
df = pd.DataFrame(rows)
df_pbar = pd.DataFrame(pbar_scot)
rows[-1]["point_size"] = 80
print(len(rows))

# 2. 将离散的 dimension 做数值映射，以便在 x 轴用散点画出，并实现抖动
dimension_unique = df["dimension"].unique()
dimension_map = {dim: i for i, dim in enumerate(dimension_unique)}
df["dimension_numeric"] = df["dimension"].map(dimension_map)

# 让每条“泳道”都有一点随机抖动（避免所有点都在一条垂直线上）
np.random.seed(42)  # 固定随机种子，便于结果可复现
jitter_strength = 0.1
df["dimension_jittered"] = df["dimension_numeric"] + np.random.uniform(
    -jitter_strength, jitter_strength, len(df)
)

plot_reason_parts_with_ci(df_pbar, "reason_parts.svg", "reason_parts.png",["#4B3F72","#468BCA","#00A9A5"])
plot_total_reason_box(df_pbar, "total_reason_box.svg", "total_reason_box.png",["#4B3F72","#468BCA","#00A9A5"])



import altair as alt

# 1) 先把“数值抖动”转为“像素抖动”
#    每个分类桶的可视宽度（像素）按经验给一个值，越大抖动越明显
bucket_px = 100  # 可微调
# df = df.copy()
# df['x_jitter_px'] = (df['dimension_jittered'] - df['dimension_numeric']) * bucket_px

rename_map = {
    'evidence_completeness': 'Completeness',
    'evidence_relevance': 'Correctness',
    'guideline_citation_accuracy': 'Consistency',
}
df = df.copy()
df['dimension_label'] = df['dimension'].map(rename_map).fillna(df['dimension'])

# x 用 dimension_label（其它逻辑不变）
x_enc = alt.X('dimension_label:N',
              title='',
              axis=alt.Axis(labelAngle=0))

y_enc = alt.Y('score:Q',
              title='Score',
              scale=alt.Scale(domainMin=3.5, nice=False),
                 # ← 取消加粗
              # 保险起见：明确不要强制从 0 开始
              axis=alt.Axis(tickMinStep=0.5,titleFontWeight='normal',titleFontSize=12))  # 可选：更密的刻度

df['model_shape'] = df['model']

color_enc = alt.Color('model:N',
                      title='',   # 想隐藏标题就用 title=None
                      scale=alt.Scale(domain=['Ours','GPT-4o','DeepSeek-R1'],
                                      range=['#4B3F72','#468BCA','#00A9A5']))

shape_enc = alt.Shape('model_shape:N',
                    #   legend=None,     # 不要 shape 的图例
                      scale=alt.Scale(domain=['Ours','GPT-4o','DeepSeek-R1'],
                                      range=['circle','square','triangle-up']))

palette = {'Ours':'#4B3F72','GPT-4o':'#468BCA','DeepSeek-R1':'#00A9A5'}
shapes  = {'Ours':'circle','GPT-4o':'square','DeepSeek-R1':'triangle-up'}

# chart = (
#     alt.Chart(df)
#     .mark_point(filled=True, stroke='white', strokeWidth=0.6, opacity=0.7)
#     .encode(
#         x=alt.X('dimension:N', title='Type of Evaluation Score',
#                 axis=alt.Axis(labelAngle=0)),
#         xOffset=alt.X('x_jitter_px:Q'),   # 用你预计算的像素级抖动
#         y=alt.Y('score:Q', title='Score'),
#         size=alt.Size('point_size:Q', legend=None,
#                       scale=alt.Scale(range=[20, 400])),
#         color=alt.Color('model:N', title=None,
#                         scale=alt.Scale(domain=list(palette.keys()),
#                                         range=list(palette.values()))),
#         shape=alt.Shape('model:N', legend=None,
#                         scale=alt.Scale(domain=list(shapes.keys()),
#                                         range=list(shapes.values()))),
#         tooltip=['model:N','patient_id:N','dimension:N','score:Q']
#     )
#     .properties(width=400, height=520, title='Scoring Distribution by Model')
#     .configure_axis(grid=True, gridDash=[2,3], labelFontSize=12, titleFontSize=14)
#     .configure_title(fontSize=16)
#     .configure_legend(labelFontSize=12)
#     .configure_view(strokeOpacity=0)
# )


chart = (
    alt.Chart(df)
    .transform_calculate(jitter='(random()-0.5)*8')  # 在像素上左右抖动（±18px）
    .mark_point(filled=True, stroke='white', strokeWidth=0.6, opacity=0.7)
    .encode(
        x=x_enc,  
        xOffset=alt.X('jitter:Q'),  # 关键：对点进行像素级偏移实现抖动
        y=y_enc,
        size=alt.Size('point_size:Q', legend=None,
                      scale=alt.Scale(range=[12, 180])),  # 调整总体大小区间
        # color=alt.Color('model:N', title=None,
        #                 scale=alt.Scale(domain=list(palette.keys()),
        #                                 range=list(palette.values()))),
        color=color_enc, 
        shape=shape_enc,
        # shape=alt.Shape('model:N', legend=None,
        #                 scale=alt.Scale(domain=list(shapes.keys()),
        #                                 range=list(shapes.values()))),
        tooltip=['model:N','patient_id:N','dimension:N','score:Q']
    )
    # .properties(width=400, height=300, title='Scoring Distribution by Model')
    .properties(width=400, height=300, title='')
    .configure_axis(grid=True, gridDash=[2,3], labelFontSize=12, titleFontSize=14)
    .configure_title(fontSize=13)
    .configure_legend(labelFontSize=14)
    .configure_view(strokeOpacity=0)
)


# --- 自定义图例：放在 chart 定义完之后、保存之前 ---
# --- 自定义图例（含两条虚线分隔三项） ---
# -------- 1) 不要在子图里用 configure_*：构建 core chart --------
# -------- 1) 不要在子图里用 configure_*：构建 core chart --------
chart_core = (
    alt.Chart(df)
    .transform_calculate(jitter='(random()-0.5)*8')
    .mark_point(filled=True, stroke='white', strokeWidth=0.6, opacity=0.7)
    .encode(
        x=x_enc,
        xOffset=alt.X('jitter:Q'),
        y=y_enc,
        size=alt.Size('point_size:Q', legend=None, scale=alt.Scale(range=[12, 180])),
        color=color_enc,         # 注意：此处 color/shape 先用 enc（稍后关图例）
        shape=shape_enc,
        tooltip=['model:N','patient_id:N','dimension:N','score:Q']
    )
    .properties(width=400, height=300, title='')
)

# 2) 关闭主图内置图例（避免与自定义图例重复）
models  = ['Ours','GPT-4o','DeepSeek-R1']
colors  = ['#4B3F72','#468BCA','#00A9A5']
shapes  = ['circle','square','triangle-up']

chart_no_legend = chart_core.encode(
    color=alt.Color('model:N', legend=None,
                    scale=alt.Scale(domain=models, range=colors)),
    shape=alt.Shape('model:N', legend=None,
                    scale=alt.Scale(domain=models, range=shapes))
)

# -------- 3) 自定义图例（含两条虚线分隔三项），宽度要与主图一致 --------
import pandas as pd
W, H = 400, 28
legend_df = pd.DataFrame({'model': models, 'xpos': [0.5, 1.5, 2.5]})
seps_df   = pd.DataFrame({'x': [1.0, 2.0]})
x_scale   = alt.Scale(domain=[0, 3])

legend_separators = (
    alt.Chart(seps_df)
    .mark_rule(size=1, strokeDash=[4,4], color='#999999')
    .encode(x=alt.X('x:Q', scale=x_scale, axis=None),
            y=alt.value(0), y2=alt.value(H))
)

legend_symbols = (
    alt.Chart(legend_df)
    .mark_point(filled=True, size=150, stroke='white', strokeWidth=0.6, opacity=0.9)
    .encode(
        x=alt.X('xpos:Q', scale=x_scale, axis=None),
        y=alt.value(H/2),
        color=alt.Color('model:N', legend=None, scale=alt.Scale(domain=models, range=colors)),
        shape=alt.Shape('model:N', legend=None, scale=alt.Scale(domain=models, range=shapes))
    )
)

legend_labels = (
    alt.Chart(legend_df)
    .mark_text(dy=10, fontSize=12)
    .encode(x=alt.X('xpos:Q', scale=x_scale, axis=None),
            y=alt.value(H/2),
            text='model')
)

legend = (legend_separators + legend_symbols + legend_labels).properties(width=W, height=H)

# -------- 4) 现在再在“最外层”设置所有 configure_* --------
final = (
    alt.vconcat(legend, chart_no_legend)
      .configure_concat(spacing=6)
      .configure_axis(grid=True, gridDash=[2,3], labelFontSize=12, titleFontSize=14,
                      titleFontWeight='normal')   # y 轴标题不加粗
      .configure_title(fontSize=13)
      .configure_view(strokeOpacity=0)
)

# -------- 5) 只保存 final，不要再保存旧的 chart --------
final.save('scatter_altair.svg')
final.save('scatter_altair.png')





# === 4) 保存最终图（保存 final，不要保存 chart_core/chart_no_legend） ===
final.save('scatter_altair.svg')
final.save('scatter_altair.png')

# （可选）快速自检
# print(type(legend_layer), type(chart_no_legend))  # 应分别是 LayerChart / Chart
# print(final.to_dict()['$schema'])                 # 建议是 .../vega-lite/v5.json





