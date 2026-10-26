import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================================
# 0. 全局学术绘图设置
# =====================================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False


def get_latest_file(pattern):
    files = glob.glob(pattern)
    return max(files, key=os.path.getmtime) if files else None


# 文件匹配模式字典
file_patterns = {
    "1W Soliton": {
        "Case A": "*Table1_cqqx_ConstantC*.csv",  # cqqx
        "Case B": "*Table1_1W_Fractional_ConstantC*.csv",  # Fractional
        "Case C": "*Table1_5thOrder_ConstantC*.csv"  # 5th Order
    },
    "2W Soliton": {
        "Case A": "*Table?_2W_cqqx_ConstantC*.csv",
        "Case B": "*Table?_2W_Fractional_ConstantC*.csv",
        "Case C": "*Table?_2W_5thOrder_ConstantC*.csv"
    }
}

# 统一的颜色映射，与折线图保持绝对一致
method_colors = {
    "Case A": "#2ca02c",  # 绿色
    "Case B": "#d62728",  # 红色
    "Case C": "#1f77b4"  # 蓝色
}

# 子图对应的指标与 Y 轴标签
metrics_map = [
    ("Loss_PDE(Total)", "$\\mathcal{L}_{\\mathrm{PDE}}$", "(a)"),
    ("Loss_q(Total)", "$\\mathcal{L}_q$", "(b)"),
    ("RMSE_q", "$\\mathrm{RMSE}_q$", "(c)"),
    ("Rel_L2_q", "$\\mathrm{Rel}-L_{2,q}$", "(d)")
]

# 🌟 目标错配量列表 (去除了基准点 -2.0)
target_a4_values = [-4.0, -3.0, -1.0, 0.0, 1.0, 2.0]

# =====================================================================
# 1. 数据提取与预处理 (提取所有样本以供箱线图统计)
# =====================================================================
all_data = []

for system, methods in file_patterns.items():
    for method, pattern in methods.items():
        file_path = get_latest_file(pattern)
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ 未找到或数据不完整: {system} - {method}")
            continue

        try:
            df = pd.read_csv(file_path)
            # 确保 Fixed_a4 列存在
            if 'Fixed_a4' in df.columns:
                # 🌟 过滤数据：只保留我们关心的错配量 (剔除 -2.0)
                # 使用 np.isclose 避免浮点数精度问题导致的遗漏
                mask = np.any([np.isclose(df['Fixed_a4'], val, atol=1e-3) for val in target_a4_values], axis=0)
                df_filtered = df[mask].copy()

                # 动态寻找初始值列作为去重依据
                init_col = next((col for col in ['Init_c', 'Init_d0', 'Init_b0'] if col in df_filtered.columns), None)
                if init_col:
                    # 保留每一个 (a4, init) 组合的最新一次实验记录
                    df_clean = df_filtered.drop_duplicates(subset=['Fixed_a4', init_col], keep='last').copy()
                else:
                    df_clean = df_filtered.copy()

                df_clean['System'] = system
                df_clean['Method'] = method
                all_data.append(df_clean)
                print(f"✅ 成功加载: {system} - {method} (样本数: {len(df_clean)})")
        except Exception as e:
            print(f"❌ 读取出错 {file_path}: {e}")

if not all_data:
    print("❌ 没有找到有效数据，退出绘图。")
    exit()

# 合并所有数据
combined_df = pd.concat(all_data, ignore_index=True)

# 确保指标列是浮点数
for metric, _, _ in metrics_map:
    if metric in combined_df.columns:
        combined_df[metric] = pd.to_numeric(combined_df[metric], errors='coerce')

# =====================================================================
# 2. 渲染 2x2 箱线图
# =====================================================================
fig, axs = plt.subplots(2, 2, figsize=(13, 9))
plt.subplots_adjust(wspace=0.22, hspace=0.2, top=0.88)  # 调整顶部空间以容纳图例

# 统一的箱体和离群点样式
flier_style = dict(marker='o', markerfacecolor='none', markeredgecolor='gray', markersize=4, alpha=0.6)
box_style = dict(edgecolor='black', alpha=0.85)

for i, (metric_col, y_label, title_label) in enumerate(metrics_map):
    row, col = divmod(i, 2)
    ax = axs[row, col]

    # 如果该指标不在数据中，跳过
    if metric_col not in combined_df.columns:
        continue

    # 去除空值以防止绘图报错
    plot_df = combined_df.dropna(subset=[metric_col])

    # 绘制箱线图
    sns.boxplot(
        data=plot_df,
        x="System",
        y=metric_col,
        hue="Method",
        ax=ax,
        palette=method_colors,
        width=0.6,
        linewidth=1.5,
        flierprops=flier_style,
        boxprops=box_style
    )

    # 设为对数坐标，展示极值差异
    ax.set_yscale('log')
    ax.grid(True, axis='y', which="major", linestyle='--', alpha=0.5, zorder=0)

    # 🌟 统一并缩短 Y 轴的显示范围，使图形更协调紧凑
    ax.set_ylim(1e-3, 1.5)

    # 标签与子图编号
    ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
    ax.set_xlabel("", fontsize=1)  # 隐藏默认的 X 轴标签 "System"

    # 将 (a)(b)(c)(d) 放在图表内部左上角
    ax.text(0.02, 0.92, title_label, transform=ax.transAxes, fontsize=18, fontweight='bold')

    # 刻度设置 (向内)
    ax.tick_params(axis='both', direction='in', labelsize=14, length=6, width=1.2)
    ax.tick_params(axis='y', which='minor', direction='in', length=3, width=1)

    # 隐藏 seaborn 自动生成的内部图例
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# =====================================================================
# 3. 创建自定义全局组合图例 (放置在图表顶部居中单行排列)
# =====================================================================
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor=method_colors["Case A"], edgecolor='black', label='Case A'),
    Patch(facecolor=method_colors["Case B"], edgecolor='black', label='Case B'),
    Patch(facecolor=method_colors["Case C"], edgecolor='black', label='Case C')
]

# 🌟 将图例居中放置在图表顶部
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.98),
           ncol=3, fontsize=15, frameon=False, columnspacing=2.0)

# =====================================================================
# 4. 保存输出
# =====================================================================
output_img = "Figure6_剔除-2 Diagnostic_BoxPlots.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()

print(f"\n🎉 2x2 对数箱线图渲染完成！已保存为高质量图片: {output_img}")