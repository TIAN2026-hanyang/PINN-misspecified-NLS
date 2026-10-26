import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================================
# 0. 全局期刊绘图样式（固定不变）
# ==========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# 固定配色（和你的参考图完全一致）
color_std = '#5B84C4'   # Standard(Tanh+Fixed LR)
color_adp = '#C95C5C'   # Fully Adaptive
color_rs  = '#4B9C94'   # Random Sampling
color_lhs = '#D87A68'   # Latin Hypercube

# ==========================================================
# 1. 读取两个CSV文件，提取RMSE_q数据
# ==========================================================
# 读取1W数据
df_1w = pd.read_csv("Table1_1W_Forward_L2_Errors_20260727_162432.csv")
# 读取2W数据
df_2w = pd.read_csv("Forward_Ablation_Results_2W_20260727_171554.csv")

# 构建字典方便取值
def build_data_dict(df):
    data_dict = {}
    for _, row in df.iterrows():
        cfg = row["Configuration"]
        samp = row["Sampling"]
        rmse = row["RMSE_q"]
        data_dict[(cfg, samp)] = rmse
    return data_dict

d1w = build_data_dict(df_1w)
d2w = build_data_dict(df_2w)

# ---------------------- 组装(a)子图数据 ----------------------
labels_a = ['1W (RS)', '1W (LHS)', '2W (RS)', '2W (LHS)']
standard_a = [
    d1w[("Tanh/fixed", "RS")],
    d1w[("Tanh/fixed", "LHS")],
    d2w[("Tanh/fixed", "RS")],
    d2w[("Tanh/fixed", "LHS")]
]
adaptive_a = [
    d1w[("Adaptive/adaptive", "RS")],
    d1w[("Adaptive/adaptive", "LHS")],
    d2w[("Adaptive/adaptive", "RS")],
    d2w[("Adaptive/adaptive", "LHS")]
]

# ---------------------- 组装(b)子图数据 ----------------------
labels_b = ['1W\n(Standard)', '1W\n(Adaptive)', '2W\n(Standard)', '2W\n(Adaptive)']
rs_b = [
    d1w[("Tanh/fixed", "RS")],
    d1w[("Adaptive/adaptive", "RS")],
    d2w[("Tanh/fixed", "RS")],
    d2w[("Adaptive/adaptive", "RS")]
]
lhs_b = [
    d1w[("Tanh/fixed", "LHS")],
    d1w[("Adaptive/adaptive", "LHS")],
    d2w[("Tanh/fixed", "LHS")],
    d2w[("Adaptive/adaptive", "LHS")]
]

# ==========================================================
# 2. 画布初始化 & 绘制双柱状子图
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(len(labels_a))
width = 0.32

# ========== 左图(a)：Standard VS Adaptive ==========
bar_std_a = ax1.bar(x - width/2, standard_a, width, label='Standard (Tanh + Fixed LR)',
                    color=color_std, edgecolor="black", linewidth=1)
bar_adp_a = ax1.bar(x + width/2, adaptive_a, width, label='Fully Adaptive (Activ + LR)',
                    color=color_adp, edgecolor="black", linewidth=1)

ax1.set_yscale("log")
ax1.set_title("(a)", fontweight="bold", fontsize=16, pad=12)
ax1.set_ylabel("RMSE (Log Scale)", fontweight="bold", fontsize=13)
ax1.set_xticks(x)
ax1.set_xticklabels(labels_a, fontsize=12)
ax1.tick_params(labelsize=11)
leg1 = ax1.legend(loc="upper left", frameon=True, fontsize=11)
leg1.get_frame().set_edgecolor("black")

# ========== 右图(b)：RS VS LHS ==========
bar_rs_b = ax2.bar(x - width/2, rs_b, width, label='RS (Random Sampling)',
                   color=color_rs, edgecolor="black", linewidth=1)
bar_lhs_b = ax2.bar(x + width/2, lhs_b, width, label='LHS (Latin Hypercube)',
                    color=color_lhs, edgecolor="black", linewidth=1)

ax2.set_yscale("log")
ax2.set_title("(b)", fontweight="bold", fontsize=16, pad=12)
ax2.set_ylabel("RMSE (Log Scale)", fontweight="bold", fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels(labels_b, fontsize=12)
ax2.tick_params(labelsize=11)
leg2 = ax2.legend(loc="upper left", frameon=True, fontsize=11)
leg2.get_frame().set_edgecolor("black")

# ==========================================================
# 3. 柱子顶部标注科学计数法数值
# ==========================================================
def add_bar_text(bar_list, ax_obj):
    for bar in bar_list:
        h_val = bar.get_height()
        ax_obj.annotate(f"{h_val:.1e}",
                        xy=(bar.get_x() + bar.get_width() / 2, h_val),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontstyle="italic")

add_bar_text(bar_std_a, ax1)
add_bar_text(bar_adp_a, ax1)
add_bar_text(bar_rs_b, ax2)
add_bar_text(bar_lhs_b, ax2)

# 总标题（和参考图顶部标题一致）
fig.suptitle("Physics-Informed Neural Network for Nonlinear Mismatch Compensation",
             fontsize=17, y=0.98)

# 布局调整 + 保存高清图片
plt.tight_layout()
plt.savefig("Ablation_1W_2W_Bar_FromCSV.png", dpi=600, bbox_inches="tight")
plt.show()
print("✅ 成功读取两个CSV文件，基于真实数据完成绘图并保存图片")