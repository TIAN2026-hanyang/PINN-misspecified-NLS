import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.ticker as ticker

# ==========================================================
# 0. 全局字体与格式设置 (顶级期刊学术规范)
# ==========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


# ==========================================================
# 1. 数据读取函数（匹配你的文件夹命名 Data_1W_Inverse_6Layer_Noise_x_Init_x.x）
# ==========================================================
def load_trajectory_data(noise, init_val, base_path=None):
    if base_path is None:
        base_path = os.path.dirname(os.path.abspath(__file__))
    folder_name = f"Data_1W_Inverse_6Layer_Noise_{noise}_Init_{init_val:.1f}"
    folder_path = os.path.join(base_path, folder_name)

    if not os.path.exists(folder_path):
        print(f"⚠️ 文件夹不存在：{folder_name}")
        return np.array([]), np.array([])

    files_in_folder = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    if not files_in_folder:
        return np.array([]), np.array([])

    target_file = next((f for f in files_in_folder if f.endswith(('.txt', '.csv', '.dat'))), None)
    if "trajectory_c.txt" in files_in_folder:
        target_file = "trajectory_c.txt"

    if target_file is None:
        return np.array([]), np.array([])

    file_path = os.path.join(folder_path, target_file)

    try:
        try:
            data = np.loadtxt(file_path, skiprows=1)
        except ValueError:
            data = np.loadtxt(file_path)

        iters = data[:, 0]
        vals = data[:, 1]
        print(f"✅ 成功读取: {folder_name}\\{target_file} (共 {len(iters)} 步)")
        return iters, vals
    except Exception as e:
        print(f"❌ 读取文件 {target_file} 失败: {e}")
        return np.array([]), np.array([])


# ==========================================================
# 自动补齐早停数据
# ==========================================================
def pad_to_max_iter(iters, vals, max_iter):
    if len(iters) > 0 and iters[-1] < max_iter:
        iters = np.append(iters, max_iter)
        vals = np.append(vals, vals[-1])
    return iters, vals


# ==========================================================
# 2. 加载数据
# ==========================================================
print("-" * 60)
print(">>> 正在扫描并加载 Init=0.0 的数据...")
iters_a_0, vals_a_0 = load_trajectory_data(noise=0, init_val=0.0)
iters_a_1, vals_a_1 = load_trajectory_data(noise=1, init_val=0.0)
iters_a_5, vals_a_5 = load_trajectory_data(noise=5, init_val=0.0)

print("\n>>> 正在扫描并加载 Init=-3.0 的数据...")
iters_b_0, vals_b_0 = load_trajectory_data(noise=0, init_val=-3.0)
iters_b_1, vals_b_1 = load_trajectory_data(noise=1, init_val=-3.0)
iters_b_5, vals_b_5 = load_trajectory_data(noise=5, init_val=-3.0)
print("-" * 60)


# ==========================================================
# 自适应放大窗口Y轴范围
# ==========================================================
def auto_zoom_ylim(axins, iters_list, vals_list, zoom_start, zoom_end):
    all_zoom_vals = []
    for itrs, vls in zip(iters_list, vals_list):
        if len(itrs) > 1:
            mask = (itrs >= zoom_start) & (itrs <= zoom_end)
            if np.any(mask):
                all_zoom_vals.extend(vls[mask])

    if all_zoom_vals:
        ymin, ymax = np.percentile(all_zoom_vals, 5), np.percentile(all_zoom_vals, 95)
        padding = (ymax - ymin) * 0.3
        if padding < 0.0015:
            padding = 0.0015
        final_ymin = ymin - padding
        final_ymax = ymax + padding
        axins.set_ylim(final_ymin, final_ymax)


# ==========================================================
# 4. 画布初始化
# ==========================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

# x轴刻度格式化：10000 → 10k
def format_func(value, tick_number):
    if value == 0:
        return "0"
    return f"{int(value / 1000)}k"
formatter = ticker.FuncFormatter(format_func)

# ===================== 左图 (a) Init = 0.0 =====================
iters_a_list = [i for i in [iters_a_0, iters_a_1, iters_a_5] if len(i) > 0]
if iters_a_list:
    max_iter_a = max([np.max(i) for i in iters_a_list])
    iters_a_0, vals_a_0 = pad_to_max_iter(iters_a_0, vals_a_0, max_iter_a)
    iters_a_1, vals_a_1 = pad_to_max_iter(iters_a_1, vals_a_1, max_iter_a)
    iters_a_5, vals_a_5 = pad_to_max_iter(iters_a_5, vals_a_5, max_iter_a)

    # 真值线：黑色点状线
    ax1.plot([0, max_iter_a], [-2.0, -2.0], color='black', linestyle=':', linewidth=2.5, label='Ground Truth (-2.0)')
    # 配色线型严格对标示例图
    if len(iters_a_0) > 0:
        ax1.plot(iters_a_0, vals_a_0, color='#348ABD', linestyle='-', linewidth=2.0, label='0% Noise (Clean)')
    if len(iters_a_1) > 0:
        ax1.plot(iters_a_1, vals_a_1, color='#8EBA42', linestyle='--', linewidth=2.0, label='1% Noise')
    if len(iters_a_5) > 0:
        ax1.plot(iters_a_5, vals_a_5, color='#E24A33', linestyle='-.', linewidth=2.0, label='5% Noise')

    ax1.set_title('(a) Parameter Convergence (1W, Init = 0.0)', fontweight='bold', fontsize=15, pad=12)
    ax1.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Recovered parameter $\\alpha_4$', fontweight='bold', fontsize=14)
    ax1.xaxis.set_major_formatter(formatter)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.tick_params(labelsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=11, edgecolor='black')

    # 左图放大区间统一修改为：10000 ~ 15000
    zoom_start_a = 10000
    zoom_end_a = 15000
    axins1 = ax1.inset_axes([0.41, 0.12, 0.54, 0.45])
    axins1.plot([0, max_iter_a], [-2.0, -2.0], color='black', linestyle=':', linewidth=2.5)
    if len(iters_a_0) > 0: axins1.plot(iters_a_0, vals_a_0, color='#348ABD', linestyle='-', linewidth=2.0)
    if len(iters_a_1) > 0: axins1.plot(iters_a_1, vals_a_1, color='#8EBA42', linestyle='--', linewidth=2.0)
    if len(iters_a_5) > 0: axins1.plot(iters_a_5, vals_a_5, color='#E24A33', linestyle='-.', linewidth=2.0)

    axins1.set_xlim(zoom_start_a, zoom_end_a)
    auto_zoom_ylim(axins1, [iters_a_0, iters_a_1, iters_a_5], [vals_a_0, vals_a_1, vals_a_5], zoom_start_a, zoom_end_a)
    axins1.xaxis.set_major_formatter(formatter)
    axins1.set_title('Zoom-in', fontweight='bold', fontsize=11)
    axins1.grid(True, linestyle=':', alpha=0.6)
    mark_inset(ax1, axins1, loc1=3, loc2=4, fc="none", ec="gray", linewidth=1.2, alpha=0.7)

# ===================== 右图 (b) Init = -3.0 =====================
iters_b_list = [i for i in [iters_b_0, iters_b_1, iters_b_5] if len(i) > 0]
if iters_b_list:
    max_iter_b = max([np.max(i) for i in iters_b_list])
    iters_b_0, vals_b_0 = pad_to_max_iter(iters_b_0, vals_b_0, max_iter_b)
    iters_b_1, vals_b_1 = pad_to_max_iter(iters_b_1, vals_b_1, max_iter_b)
    iters_b_5, vals_b_5 = pad_to_max_iter(iters_b_5, vals_b_5, max_iter_b)

    ax2.plot([0, max_iter_b], [-2.0, -2.0], color='black', linestyle=':', linewidth=2.5, label='Ground Truth (-2.0)')
    if len(iters_b_0) > 0:
        ax2.plot(iters_b_0, vals_b_0, color='#348ABD', linestyle='-', linewidth=2.0, label='0% Noise (Clean)')
    if len(iters_b_1) > 0:
        ax2.plot(iters_b_1, vals_b_1, color='#8EBA42', linestyle='--', linewidth=2.0, label='1% Noise')
    if len(iters_b_5) > 0:
        ax2.plot(iters_b_5, vals_b_5, color='#E24A33', linestyle='-.', linewidth=2.0, label='5% Noise')

    ax2.set_title('(b) Parameter Convergence (1W, Init = -3.0)', fontweight='bold', fontsize=15, pad=12)
    ax2.set_xlabel('Iterations', fontweight='bold', fontsize=14)
    ax2.xaxis.set_major_formatter(formatter)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.tick_params(labelsize=12)
    ax2.legend(loc='lower right', frameon=True, fontsize=11, edgecolor='black')

    # 右图放大区间：10000 ~ 15000
    zoom_start_b = 10000
    zoom_end_b = 15000
    axins2 = ax2.inset_axes([0.36, 0.48, 0.60, 0.44])
    axins2.plot([0, max_iter_b], [-2.0, -2.0], color='black', linestyle=':', linewidth=2.5)
    if len(iters_b_0) > 0: axins2.plot(iters_b_0, vals_b_0, color='#348ABD', linestyle='-', linewidth=2.0)
    if len(iters_b_1) > 0: axins2.plot(iters_b_1, vals_b_1, color='#8EBA42', linestyle='--', linewidth=2.0)
    if len(iters_b_5) > 0: axins2.plot(iters_b_5, vals_b_5, color='#E24A33', linestyle='-.', linewidth=2.0)

    axins2.set_xlim(zoom_start_b, zoom_end_b)
    auto_zoom_ylim(axins2, [iters_b_0, iters_b_1, iters_b_5], [vals_b_0, vals_b_1, vals_b_5], zoom_start_b, zoom_end_b)
    axins2.xaxis.set_major_formatter(formatter)
    axins2.set_title('Zoom-in', fontweight='bold', fontsize=11)
    axins2.grid(True, linestyle=':', alpha=0.6)
    mark_inset(ax2, axins2, loc1=1, loc2=2, fc="none", ec="gray", linewidth=1.2, alpha=0.7)

# 保存高清图片
plt.tight_layout(pad=2.0)
output_filename = "Figure_1w_Convergence.png"
plt.savefig(output_filename, dpi=600, bbox_inches='tight')
print(f"✅ 两张子图放大区间全部修改为 10000~15000，图片已保存")
plt.show()