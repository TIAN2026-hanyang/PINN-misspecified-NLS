import deepxde as dde
import numpy as np
import os
import csv
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================================
# 0. 全局设置与实验配置
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前计算设备: {device} | 模式: 1W 正问题实验 (6层网络 + 2万步迭代 | Tanh vs 自适应激活 对照)")

# 2×2 对照组：激活函数 (Tanh固定 / 自适应) × 采样方式 (随机RS / LHS)
experiments = [
    {"name": "Tanh_RS",        "is_adaptive": False, "sampling": "Random"},
    {"name": "Adaptive_RS",    "is_adaptive": True,  "sampling": "Random"},
    {"name": "Tanh_LHS",       "is_adaptive": False, "sampling": "LHS"},
    {"name": "Adaptive_LHS",   "is_adaptive": True,  "sampling": "LHS"}
]

# 🛡️ 安全命名：加上时间戳，防止覆盖历史宝贵数据
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"Table1_1W_Forward_L2_Errors_{timestamp}.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Configuration", "Sampling", "Total_Steps",
        "Rel_L2_q", "RMSE_q", "Rel_L2_u", "Rel_L2_v",
        "Final_Total_Loss", "L_PDE_real", "L_PDE_imag"
    ])

# ==========================================================
# 1. 1W 物理参数与精确解
# ==========================================================
alpha1, alpha2, alpha3, alpha4 = 0.4, 1.0, -1.0, -2.0

def compute_k(alpha4_val):
    disc = alpha3 ** 2 - 4 * alpha2 * (alpha1 + 0.5 * alpha4_val)
    return (-alpha3 - np.sqrt(disc)) / (2 * alpha2)

def exact_solution(xyt):
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    k = compute_k(alpha4)
    # 完美保留你原版的正确公式: 8 * k * x * y
    denom = 4 * k ** 2 * y ** 2 + 4 * x ** 2 + 8 * k * x * y + 4 * alpha4 ** 2 * t ** 2 + 1
    amp = 4 * (1 - 2j * alpha4 * t) / denom - 1
    phase = np.exp(-1j * alpha4 * t)
    q = amp * phase
    return np.hstack([np.real(q), np.imag(q)])

# ==========================================================
# 🚀 自动化大循环
# ==========================================================
for idx, exp in enumerate(experiments):
    exp_name = exp["name"]
    is_adaptive = exp["is_adaptive"]
    sampling_method = exp["sampling"]
    print(f"\n{'=' * 75}\n▶️ 正在执行 1W 正问题实验 {idx + 1}/4: {exp_name}\n{'=' * 75}")

    # 锁定随机种子保证可复现
    dde.config.set_random_seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)

    # 1W 孤子的标准时空域
    x_min, x_max, y_min, y_max, t_min, t_max = -10, 10, -10, 10, 0.0, 1.0
    geom = dde.geometry.Rectangle([x_min, y_min], [x_max, y_max])
    timedomain = dde.geometry.TimeDomain(t_min, t_max)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    def on_initial(xyt, on_boundary):
        return np.isclose(xyt[2], t_min)

    def is_on_boundary(xyt, on_boundary_flag):
        return on_boundary_flag and (np.isclose(xyt[0], x_min) or np.isclose(xyt[0], x_max) or
                                     np.isclose(xyt[1], y_min) or np.isclose(xyt[1], y_max))

    ic_real = dde.icbc.IC(geomtime, lambda x: exact_solution(x)[:, 0:1], on_initial, component=0)
    ic_imag = dde.icbc.IC(geomtime, lambda x: exact_solution(x)[:, 1:2], on_initial, component=1)
    bc_real = dde.icbc.DirichletBC(geomtime, lambda x: exact_solution(x)[:, 0:1], is_on_boundary, component=0)
    bc_imag = dde.icbc.DirichletBC(geomtime, lambda x: exact_solution(x)[:, 1:2], is_on_boundary, component=1)

    # 激活函数配置判断
    if is_adaptive:
        a_param = torch.tensor([0.1], requires_grad=True, device=device)
        def adaptive_tanh(x):
            return torch.tanh(10.0 * a_param * x)
        activation = adaptive_tanh
        ext_vars = [a_param]
    else:
        activation = "tanh"
        ext_vars = None

    # ==========================================================
    # 🌟 正问题 PDE: 没有错配，没有补偿项
    # ==========================================================
    def pde(xyt, q):
        u, v = q[:, 0:1], q[:, 1:2]
        u_t = dde.grad.jacobian(q, xyt, i=0, j=2)
        v_t = dde.grad.jacobian(q, xyt, i=1, j=2)
        v_xx = dde.grad.hessian(q, xyt, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(q, xyt, component=1, i=1, j=1)
        v_xy = dde.grad.hessian(q, xyt, component=1, i=0, j=1)
        u_xx = dde.grad.hessian(q, xyt, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(q, xyt, component=0, i=1, j=1)
        u_xy = dde.grad.hessian(q, xyt, component=0, i=0, j=1)

        q_norm_sq = u ** 2 + v ** 2

        # Eq. 4a: F_R (实部残差, 首项严格为 -v_t)
        F_R = -v_t + alpha1 * u_xx + alpha2 * u_yy + alpha3 * u_xy - alpha4 * q_norm_sq * u
        # Eq. 4b: F_I (虚部残差, 首项严格为 u_t)
        F_I = u_t + alpha1 * v_xx + alpha2 * v_yy + alpha3 * v_xy - alpha4 * q_norm_sq * v

        return [F_R, F_I]

    loss_weights_list = [1, 1, 1, 1, 1, 1]

    # 严格遵照论文采样点个数 N_F=10000, N_B=2000, N_I=2000
    dist_type = "LHS" if sampling_method == "LHS" else "pseudo"
    data = dde.data.TimePDE(
        geomtime, pde, [ic_real, ic_imag, bc_real, bc_imag],
        num_domain=10000, num_boundary=2000, num_initial=2000, num_test=5000
    )
    data.train_distribution = dist_type
    data.resample_train_points()

    # 🚀 网络配置：6个隐藏层，每层 40 宽
    net = dde.nn.FNN([3] + [40] * 6 + [2], activation, "Glorot normal")
    model = dde.Model(data, net)

    class ExponentialLR(dde.callbacks.Callback):
        def __init__(self):
            super().__init__()
        def on_epoch_end(self):
            current_step = self.model.train_state.step
            for pg in self.model.opt.param_groups:
                pg['lr'] = 1e-3 * (0.9 ** (current_step / 2000))

    # 只有自适应配置才应用学习率衰减
    callbacks_list = [ExponentialLR()] if is_adaptive else []

    # 🚀 Phase 1: Adam 10000 步
    print(">>> Phase 1: Adam Training (10000 steps)...")
    model.compile("adam", lr=1e-3, loss_weights=loss_weights_list, external_trainable_variables=ext_vars)
    losshistory, train_state = model.train(iterations=10000, display_every=5000, callbacks=callbacks_list)

    # 🚀 Phase 2: L-BFGS 10000 步
    print(">>> Phase 2: L-BFGS Fine-tuning (max 10000 steps)...")
    dde.optimizers.config.set_LBFGS_options(maxiter=10000, maxfun=10000, maxcor=50, ftol=1.0 * np.finfo(float).eps,
                                            gtol=1e-5, maxls=50)
    model.compile("L-BFGS", loss_weights=loss_weights_list, external_trainable_variables=ext_vars)
    losshistory_lbfgs, train_state2 = model.train(display_every=5000)

    # ==========================================================
    # 📈 模块 A: 绘制双阶段 Loss 对数曲线图
    # ==========================================================
    history = losshistory_lbfgs if losshistory_lbfgs is not None else losshistory
    steps = np.array(history.steps)
    loss_train = np.array(history.loss_train)
    total_loss_array = np.sum(loss_train, axis=1)

    # 🚀 对齐画图掩码：前 10000 步是 Adam，后面是 L-BFGS
    adam_mask = steps <= 10000
    lbfgs_mask = steps >= 10000

    plt.figure(figsize=(12, 5))
    plt.rcParams['font.family'] = 'serif'

    plt.subplot(1, 2, 1)
    plt.semilogy(steps[adam_mask], total_loss_array[adam_mask], 'b-', linewidth=1.5, label='Total Loss')
    plt.title(f'(e) iterations (Adam) - {exp_name}')
    plt.xlabel('Iterations');
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6);
    plt.legend()

    if np.any(lbfgs_mask):
        plt.subplot(1, 2, 2)
        plt.semilogy(steps[lbfgs_mask], total_loss_array[lbfgs_mask], 'b-', linewidth=1.5, label='Total Loss')
        plt.title(f'(f) iterations (L-BFGS) - {exp_name}')
        plt.xlabel('Iterations');
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6);
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"Loss_Curve_{exp_name}.png", dpi=300)
    plt.close()
    print(f"📊 Loss 曲线已保存为: Loss_Curve_{exp_name}.png")

    # ==========================================================
    # 📐 模块 B: 计算相对 L2 范数误差 + RMSE
    # ==========================================================
    loss_pde_real = loss_train[-1, 0]
    loss_pde_imag = loss_train[-1, 1]
    loss_total = total_loss_array[-1]

    # 🌟 提取真实迭代步数 (对齐 10000 步切换点)
    total_iters = train_state2.step if train_state2.step > 10000 else train_state.step

    x_eval = np.linspace(x_min, x_max, 100)
    y_eval = np.linspace(y_min, y_max, 100)
    t_eval = np.linspace(t_min, t_max, 21)
    X, Y, T = np.meshgrid(x_eval, y_eval, t_eval)
    xyt_eval = np.hstack((X.ravel()[:, None], Y.ravel()[:, None], T.ravel()[:, None]))

    q_exact = exact_solution(xyt_eval)
    q_pred = model.predict(xyt_eval)

    u_exact, v_exact = q_exact[:, 0], q_exact[:, 1]
    u_pred, v_pred = q_pred[:, 0], q_pred[:, 1]

    rel_L2_u = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    rel_L2_v = np.linalg.norm(v_pred - v_exact) / np.linalg.norm(v_exact)

    norm_diff_q = np.sqrt(np.sum((u_pred - u_exact) ** 2 + (v_pred - v_exact) ** 2))
    norm_exact_q = np.sqrt(np.sum(u_exact ** 2 + v_exact ** 2))
    rel_L2_q = norm_diff_q / norm_exact_q

    rmse_q = np.sqrt(np.mean((u_pred - u_exact) ** 2 + (v_pred - v_exact) ** 2))

    config_str = "Adaptive/adaptive" if is_adaptive else "Tanh/fixed"
    dist_str = "RS" if sampling_method == "Random" else "LHS"

    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            config_str,
            dist_str,
            total_iters,
            f"{rel_L2_q:.3e}",
            f"{rmse_q:.3e}",
            f"{rel_L2_u:.3e}",
            f"{rel_L2_v:.3e}",
            f"{loss_total:.3e}",
            f"{loss_pde_real:.3e}",
            f"{loss_pde_imag:.3e}"
        ])

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"✅ 实验 {idx + 1}/4 完成！q(x,t) 相对误差: {rel_L2_q:.3e} | RMSE: {rmse_q:.3e}")

print(f"\n🎉 1W 正问题四组基准实验 (Tanh vs 自适应激活) 已全部跑完！请查收: {csv_file}")