import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =====================================================================
# 0. 全局学术绘图设置
# =====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


# =====================================================================
# 1. 文件匹配模式与面板配置
# =====================================================================
def get_latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# 🌟 恢复为最具物理意义的长标题
panels_config = {
    "(a) Case A": {
        "1W": "*Table1_cqqx_ConstantC*.csv",
        "2W": "*Table?_2W_cqqx_ConstantC*.csv"
    },
    "(b) Case B": {
        "1W": "*Table1_1W_Fractional_ConstantC*.csv",
        "2W": "*Table?_2W_Fractional_ConstantC*.csv"
    },
    "(c) Case C": {
        "1W": "*Table1_5thOrder_ConstantC*.csv",
        "2W": "*Table?_2W_5thOrder_ConstantC*.csv"
    }
}

# 🌟 为三个不同的物理扰动分别定制高级撞色方案
colors = {
    "(a) Case A": {"1W": "#2ca02c", "2W": "#9467bd"},  # 绿 vs 紫
    "(b) Case B": {"1W": "#f39c12", "2W": "#e74c3c"},  # 亮黄 vs 橘红
    "(c) Case C": {"1W": "#1f77b4", "2W": "#d62728"}  # 经典蓝 vs 深红
}

# 线型与标记对齐：1W(实线, 圆点) vs 2W(虚线, 方块)
linestyles = {"1W": "-", "2W": "--"}
markers = {"1W": "o", "2W": "s"}

# 目标 X 轴的物理错配量
target_a4_values = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]

# =====================================================================
# 2. 核心大循环：数据读取与 1x3 渲染
# =====================================================================
# 创建 1x3 子图
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), sharey=True)
plt.subplots_adjust(wspace=0.1)

# 动态寻找代表 Recovered Coefficient 的列名
valid_coeff_cols = ['Recovered_c', 'Rec_c', 'Rec_d0', 'Rec_a0']

for i, (panel_title, systems) in enumerate(panels_config.items()):
    ax = axs[i]

    # 绘制坐标辅助线 (Ground Truth 与 y=0 基准线)
    ax.axvline(x=-2.0, color='gray', linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)
    ax.axhline(y=0.0, color='gray', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)

    for system_name, pattern in systems.items():
        file_path = get_latest_file(pattern)
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ 未找到文件跳过: {panel_title} - {system_name} ({pattern})")
            continue

        try:
            df = pd.read_csv(file_path)
            # 确保 Fixed_a4 存在
            if 'Fixed_a4' not in df.columns:
                continue

            # 寻找恢复系数列
            target_col = next((col for col in valid_coeff_cols if col in df.columns), None)
            if not target_col:
                print(f"⚠️ 找不到有效的数据列: {file_path}")
                continue

            # 提取每一个 a4 下的 Mean, Min, Max 以及原始散点
            grouped = df.groupby('Fixed_a4')[target_col]

            x_vals = []
            means = []
            yerr_lower = []
            yerr_upper = []

            # 用于绘制独立散点
            scatter_x = []
            scatter_y = []

            for a4 in target_a4_values:
                if a4 in grouped.groups:
                    group_data = grouped.get_group(a4).dropna().values
                    if len(group_data) > 0:
                        mean_val = np.mean(group_data)
                        min_val = np.min(group_data)
                        max_val = np.max(group_data)

                        x_vals.append(a4)
                        means.append(mean_val)
                        # Error bar 需要的偏差量
                        yerr_lower.append(mean_val - min_val)
                        yerr_upper.append(max_val - mean_val)

                        # 为了防止散点和均值点重合，添加轻微的 X 轴偏移 (Jitter)
                        jitter = -0.05 if system_name == "1W" else 0.05
                        scatter_x.extend([a4 + jitter] * len(group_data))
                        scatter_y.extend(group_data)

            if x_vals:
                # 获取该子图对应的颜色与线型
                color = colors[panel_title][system_name]
                ls = linestyles[system_name]
                marker = markers[system_name]

                # 1. 绘制带误差棒的折线 (Mean, Min, Max)
                ax.errorbar(x_vals, means, yerr=[yerr_lower, yerr_upper],
                            fmt=marker, color=color, linestyle=ls, linewidth=2.0,
                            markersize=7, capsize=4, capthick=1.5, alpha=0.9, zorder=4)

                # 2. 在背景绘制带透明度的独立散点 (Individual initialization)
                ax.scatter(scatter_x, scatter_y, color=color, alpha=0.3, s=30, edgecolors='none', zorder=3)

        except Exception as e:
            print(f"❌ 读取出错 {file_path}: {e}")

    # =================子图细节美化=================
    ax.set_title(panel_title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("$\\alpha_4^{\\mathrm{fixed}}$", fontsize=14, fontweight='bold')

    # tong'y's 🌟 恢复完整的纵坐标名称
    if i == 0:
        ax.set_ylabel("$c$", fontsize=15, fontweight='bold')

    ax.set_xticks(target_a4_values)
    ax.tick_params(axis='both', direction='in', labelsize=13, length=5, width=1.0)

    # 网格线设置
    ax.grid(True, which="major", linestyle=':', alpha=0.6, zorder=0)

# =====================================================================
# 3. 顶部全局统合图例 (采用纯黑灰指示线型，忽略具体颜色)
# =====================================================================
global_legend_elements = [
    Line2D([0], [0], color='black', lw=2, linestyle='-', marker='o', markersize=7, label='1W soliton'),
    Line2D([0], [0], color='black', lw=2, linestyle='--', marker='s', markersize=7, label='2W soliton'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', alpha=0.5, markersize=8,
           label='Individual init.'),
    Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='Ground Truth')
]

# 将图例仅放置在第一个子图 (图a) 的内部左上角空日处
axs[0].legend(handles=global_legend_elements, loc='upper left',
              ncol=1, fontsize=12, framealpha=0.95, edgecolor='gray', borderpad=0.6)

# =====================================================================
# 4. 保存输出
# =====================================================================
output_img = "Figure5 图例_Recovered_Coeff_1x3.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n🎉 1x3 带误差棒折线图生成完成！已保存为高清图片: {output_img}")