import numpy as np
import deepxde as dde
import os
import csv
import torch
import itertools
from scipy.stats import qmc
from datetime import datetime
import matplotlib.pyplot as plt

# ==========================================================
# 0. 全局设置
# ==========================================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
dde.config.set_random_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {device} | 任务: 【最强对照组】2W cqqx 常系数代偿 + 10万点 LHS 全域评估 + 21组全量大名单")

# ==========================================================
# 1. 实验配置 (🌟 21组全量大名单)
# ==========================================================
alpha4_options = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0]
c_init_options = [-1.0, 0.0, 1.0]
experiment_configs = list(itertools.product(alpha4_options, c_init_options))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"Table1_2W_cqqx_ConstantC_LHS_All21_{timestamp}.csv"

# 🌟 严格对齐导师要求的五大核心指标
with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Fixed_a4", "Init_c", "Recovered_c",
        "Loss_PDE(Total)", "Loss_q(Total)", "Global_PDE_RMSE", "Eq17_R_mis",
        "RMSE_q", "Rel_L2_q", "Total_Steps"
    ])

print(f"🚀 共计 {len(experiment_configs)} 组实验准备就绪！全量数据表将保存至: {csv_file}")


# ----------------------------------------------------------
# 🌟 Callbacks (包含全套监控)
# ----------------------------------------------------------
class ExponentialLR(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self):
        current_step = self.model.train_state.step
        for pg in self.model.opt.param_groups:
            pg['lr'] = 1e-3 * (0.9 ** (current_step / 2000))


class VarLogger(dde.callbacks.Callback):
    def __init__(self, target_var, var_name, a_param):
        super().__init__()
        self.target_var = target_var
        self.var_name = var_name
        self.a_param = a_param

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % 5000 == 0 or step <= 10:
            try:
                val = self.target_var.item()
                if step % 5000 == 0:
                    print(
                        f"    [Step {step:05d}] 🔮 预测 {self.var_name} = {val:.6f} | a_param={self.a_param.item():.4f}")
            except:
                pass


class LossWeightDecay(dde.callbacks.Callback):
    def __init__(self, lambda_d0=100.0, lambda_d_min=10.0, tau=2000.0):
        super().__init__()
        self.lambda_d0 = lambda_d0
        self.lambda_d_min = lambda_d_min
        self.tau = tau

    def on_epoch_end(self):
        step = self.model.train_state.step
        lam = self.lambda_d_min + (self.lambda_d0 - self.lambda_d_min) * np.exp(-step / self.tau)
        if self.model.loss_weights is not None:
            new_w = list(self.model.loss_weights)
            new_w[6], new_w[7] = lam, lam
            self.model.loss_weights = new_w
        if step % 5000 == 0 or step <= 10:
            print(f"    [Step {step:05d}] 📉 动态数据权重 λ_d = {lam:.2f}")


class MonitorLogger(dde.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % 5000 == 0 or step <= 10:
            loss = self.model.train_state.loss_train
            w = self.model.loss_weights
            l_phys = w[0] * loss[0] + w[1] * loss[1]
            l_data = w[6] * loss[6] + w[7] * loss[7]
            if l_phys > 1e-12:
                print(f"    [Step {step:05d}] 📊 实时平衡比 R(k) = {l_data / l_phys:.2f}")


# ==========================================================
# 2. 核心大循环
# ==========================================================
for exp_idx, (current_alpha4, current_c_guess) in enumerate(experiment_configs, 1):
    dde.config.set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'=' * 70}")
    print(
        f"▶️ 对照组测试 {exp_idx}/{len(experiment_configs)} | 固定 a4: {current_alpha4:.1f} | 初始 c: {current_c_guess:.1f}")
    print(f"{'=' * 70}\n")

    alpha1, alpha2, alpha3, alpha4_true = 0.4, 1.0, -1.0, -2.0
    alpha4_fixed = current_alpha4

    # 🌟 恢复为纯粹的常系数 c_var
    c_var = dde.Variable(current_c_guess)


    def compute_k(alpha4_val):
        disc = alpha3 ** 2 - 4 * alpha2 * (alpha1 + 0.5 * alpha4_val)
        return (-alpha3 - np.sqrt(disc)) / (2 * alpha2)


    # 🌟🌟🌟 替换为 2W 怪波解 🌟🌟🌟
    def exact_solution(xyt):
        x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
        k = compute_k(alpha4_true)
        a4 = alpha4_true

        G21 = (
                    -192 * k ** 4 * y ** 4 - 1152 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 - 960 * a4 ** 4 * t ** 4 - 768 * k ** 3 * x * y ** 3 - 2304 * k * a4 ** 2 * t ** 2 * x * y - 1152 * k ** 2 * x ** 2 * y ** 2 - 1152 * a4 ** 2 * t ** 2 * x ** 2 - 768 * k * x ** 3 * y - 288 * k ** 2 * y ** 2 - 864 * a4 ** 2 * t ** 2 - 192 * x ** 4 - 576 * k * x * y - 288 * x ** 2 + 36)
        G22 = (
                    -384 * k ** 4 * y ** 4 - 768 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 - 384 * a4 ** 4 * t ** 4 - 1536 * k ** 3 * x * y ** 3 - 1536 * k * a4 ** 2 * t ** 2 * x * y - 2304 * k ** 2 * x ** 2 * y ** 2 - 768 * a4 ** 2 * t ** 2 * x ** 2 - 1536 * k * x ** 3 * y + 576 * k ** 2 * y ** 2 - 192 * a4 ** 2 * t ** 2 - 384 * x ** 4 + 1152 * k * x * y + 576 * x ** 2 + 360)
        H2 = (
                    64 * k ** 6 * y ** 6 + 192 * k ** 4 * a4 ** 2 * t ** 2 * y ** 4 + 192 * k ** 2 * a4 ** 4 * t ** 4 * y ** 2 + 64 * a4 ** 6 * t ** 6 + 384 * k ** 5 * x * y ** 5 + 768 * k ** 3 * a4 ** 2 * t ** 2 * x * y ** 3 + 384 * k * a4 ** 4 * t ** 4 * x * y + 960 * k ** 4 * x ** 2 * y ** 4 + 1152 * k ** 2 * a4 ** 2 * t ** 2 * x ** 2 * y ** 2 + 192 * a4 ** 4 * t ** 4 * x ** 2 + 1280 * k ** 3 * x ** 3 * y ** 3 + 768 * k * a4 ** 2 * t ** 2 * x ** 3 * y + 48 * k ** 4 * y ** 4 - 288 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 + 960 * k ** 2 * x ** 4 * y ** 2 + 432 * a4 ** 4 * t ** 4 + 192 * a4 ** 2 * t ** 2 * x ** 4 + 192 * k ** 3 * x * y ** 3 - 576 * k * a4 ** 2 * t ** 2 * x * y + 384 * k * x ** 5 * y + 288 * k ** 2 * x ** 2 * y ** 2 - 288 * a4 ** 2 * t ** 2 * x ** 2 + 64 * x ** 6 + 192 * k * x ** 3 * y + 108 * k ** 2 * y ** 2 + 396 * a4 ** 2 * t ** 2 + 48 * x ** 4 + 216 * k * x * y + 108 * x ** 2 + 9)

        amp_real = 1 + (G21 / H2)
        amp_imag = (-a4 * G22 * t) / H2
        q_real = amp_real * np.cos(-a4 * t) - amp_imag * np.sin(-a4 * t)
        q_imag = amp_real * np.sin(-a4 * t) + amp_imag * np.cos(-a4 * t)
        return np.hstack([q_real, q_imag])


    # ==========================================================
    # 🌟 常系数版 PDE (已修改为 cqqx 扰动项)
    # ==========================================================
    def pde(xyt, q):
        u, v = q[:, 0:1], q[:, 1:2]
        u_t = dde.grad.jacobian(q, xyt, i=0, j=2)
        v_t = dde.grad.jacobian(q, xyt, i=1, j=2)

        u_xx = dde.grad.hessian(q, xyt, component=0, i=0, j=0)
        u_yy = dde.grad.hessian(q, xyt, component=0, i=1, j=1)
        u_xy = dde.grad.hessian(q, xyt, component=0, i=0, j=1)

        v_xx = dde.grad.hessian(q, xyt, component=1, i=0, j=0)
        v_yy = dde.grad.hessian(q, xyt, component=1, i=1, j=1)
        v_xy = dde.grad.hessian(q, xyt, component=1, i=0, j=1)

        # 🌟 增加对 x 的一阶偏导数
        u_x = dde.grad.jacobian(q, xyt, i=0, j=0)
        v_x = dde.grad.jacobian(q, xyt, i=1, j=0)

        q_norm_sq = u ** 2 + v ** 2

        # 🚀 替换为 cqqx 扰动项 (即 c_var * q * q_x)
        perturb_u = c_var * (u * u_x - v * v_x)
        perturb_v = c_var * (u * v_x + v * u_x)

        # 完美对齐的控制方程
        F_R = -v_t + alpha1 * u_xx + alpha2 * u_yy + alpha3 * u_xy - alpha4_fixed * q_norm_sq * u + perturb_u
        F_I = u_t + alpha1 * v_xx + alpha2 * v_yy + alpha3 * v_xy - alpha4_fixed * q_norm_sq * v + perturb_v

        return [F_R, F_I]


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

    obs_num = 1000
    obs_pts = geomtime.random_points(obs_num)
    obs_data = exact_solution(obs_pts)
    observe_u = dde.icbc.PointSetBC(obs_pts, obs_data[:, 0:1], component=0)
    observe_v = dde.icbc.PointSetBC(obs_pts, obs_data[:, 1:2], component=1)

    data = dde.data.TimePDE(
        geomtime, pde, [ic_real, ic_imag, bc_real, bc_imag, observe_u, observe_v],
        num_domain=10000, num_boundary=2000, num_initial=2000, anchors=obs_pts, num_test=5000
    )
    data.train_distribution = "LHS"
    data.resample_train_points()

    a_param = torch.tensor([0.1], requires_grad=True, device=device)


    def adaptive_tanh(x):
        return torch.tanh(10.0 * a_param * x)


    net = dde.nn.FNN([3] + [40] * 6 + [2], adaptive_tanh, "Glorot normal")
    model = dde.Model(data, net)

    # ==========================================================
    # 3. 训练过程
    # ==========================================================
    loss_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0]
    train_vars = [c_var, a_param]

    print(">>> Phase 1: Adam Training...")
    model.compile("adam", lr=1e-3, loss_weights=loss_w, external_trainable_variables=train_vars)

    lr_cb = ExponentialLR()
    var_cb = VarLogger(c_var, "c", a_param)
    weight_decay_cb = LossWeightDecay(lambda_d0=100.0, lambda_d_min=10.0, tau=2000.0)
    monitor_cb = MonitorLogger()
    losshistory, train_state = model.train(iterations=10000, display_every=5000,
                                           callbacks=[var_cb, lr_cb, weight_decay_cb, monitor_cb])

    print(">>> Phase 2: L-BFGS Fine-tuning...")
    dde.optimizers.config.set_LBFGS_options(maxiter=10000, maxfun=10000, maxcor=50, ftol=1.0 * np.finfo(float).eps,
                                            gtol=1e-5, maxls=50)
    model.compile("L-BFGS", loss_weights=model.loss_weights, external_trainable_variables=train_vars)
    losshistory2, train_state2 = model.train(display_every=5000, callbacks=[var_cb, monitor_cb])

    # ==========================================================
    # 4. 指标精密计算 (LHS 10 万点独立测评)
    # ==========================================================
    try:
        final_c = float(c_var.detach().cpu().numpy()) if hasattr(c_var, 'detach') else float(c_var.numpy())
    except:
        final_c = current_c_guess

    total_iters = train_state2.step if train_state2.step > 10000 else train_state.step

    history_for_loss = losshistory2 if losshistory2 is not None else losshistory
    loss_train_array = np.array(history_for_loss.loss_train)
    # 🌟 合并实部和虚部的 PDE Loss 得到 Total Loss_PDE
    loss_pde_total = loss_train_array[-1, 0] + loss_train_array[-1, 1]
    # 🌟 合并关于 q 的训练集 Loss (包含初值、边界和 1000 个观测点) 得到 Loss_q(Total)
    loss_q_total = np.sum(loss_train_array[-1, 2:])

    print("\n>>> 开始生成 100,000 个 LHS 独立测试点进行物理评估...")
    n_global_test = 100000
    lhs_sampler = qmc.LatinHypercube(d=3, seed=seed + 2026)
    lhs_unit = lhs_sampler.random(n=n_global_test)
    xyt_eval = qmc.scale(lhs_unit, l_bounds=[x_min, y_min, t_min], u_bounds=[x_max, y_max, t_max]).astype(np.float32)

    # 🌟 修复处：补齐了 rmse_q 这一行！
    q_exact = exact_solution(xyt_eval)
    q_pred = model.predict(xyt_eval)
    err_sq = (q_pred[:, 0] - q_exact[:, 0]) ** 2 + (q_pred[:, 1] - q_exact[:, 1]) ** 2
    rmse_q = np.sqrt(np.mean(err_sq))
    rel_L2_q = np.sqrt(np.sum(err_sq)) / np.sqrt(np.sum(q_exact[:, 0] ** 2 + q_exact[:, 1] ** 2))

    # 【指标2】全域 PDE 残差 RMSE (常系数模式)
    pde_res = model.predict(xyt_eval, operator=pde)
    pde_sq = (np.asarray(pde_res[0]).reshape(-1) ** 2 + np.asarray(pde_res[1]).reshape(-1) ** 2)
    global_pde_rmse = np.sqrt(np.mean(pde_sq))


    # ==========================================================
    # 🌟 【指标3】公式(17) 失配残差 R_mis (修改为 cqqx版)
    # ==========================================================
    def eq17_operator(xyt, q):
        u, v = q[:, 0:1], q[:, 1:2]

        # 🌟 增加对 x 的一阶偏导数
        u_x = dde.grad.jacobian(q, xyt, i=0, j=0)
        v_x = dde.grad.jacobian(q, xyt, i=1, j=0)

        q_norm_sq = u ** 2 + v ** 2
        delta_a4 = alpha4_fixed - alpha4_true

        # 🚀 替换为 cqqx 扰动项
        perturb_u = c_var * (u * u_x - v * v_x)
        perturb_v = c_var * (u * v_x + v * u_x)

        R_mis_R = -delta_a4 * q_norm_sq * u + perturb_u
        R_mis_I = -delta_a4 * q_norm_sq * v + perturb_v
        return [R_mis_R, R_mis_I]


    R_mis_res = model.predict(xyt_eval, operator=eq17_operator)
    R_mis_sq = (np.asarray(R_mis_res[0]).reshape(-1) ** 2 + np.asarray(R_mis_res[1]).reshape(-1) ** 2)
    eq17_R_mis_rmse = np.sqrt(np.mean(R_mis_sq))

    # 写入结果 (保留原有的 10 列)
    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            f"{alpha4_fixed:.1f}", f"{current_c_guess:.1f}", f"{final_c:.6f}",
            f"{loss_pde_total:.2e}", f"{loss_q_total:.2e}", f"{global_pde_rmse:.2e}", f"{eq17_R_mis_rmse:.2e}",
            f"{rmse_q:.2e}", f"{rel_L2_q:.2e}",
            total_iters
        ])

    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(
        f"✅ 第 {exp_idx}/21 组对照组跑完！Rec_c: {final_c:.6f} | 公式17: {eq17_R_mis_rmse:.2e} | Global_PDE: {global_pde_rmse:.2e}")

print(f"\n🎉 完美！21组全量“cqqx 常系数”对照测试运行结束！数据表已保存至: {csv_file}")