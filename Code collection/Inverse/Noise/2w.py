import deepxde as dde
import numpy as np
import os
import csv
import torch
from datetime import datetime

# ==========================================================
# 0. 全局设置与种子 (绝对一致性保证)
# ==========================================================
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
dde.config.set_random_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 计算设备: {device} | 任务: 2W 单参数反问题 (6层40宽 + 1万Adam+1万LBFGS + 动态权重衰减)")

# ==========================================================
# 1. 批量实验配置
# ==========================================================
noise_levels = [0.00, 0.01, 0.05]
alpha4_inits = [0.0, -1.0, -3.0]

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = f"Table3_2W_Inverse_alpha4_6Layer_DynamicWeight_{timestamp}.csv"

with open(csv_file, mode='w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Noise", "Init_alpha4", "Recovered_a4", "E_a4",
        "IC_PDE_Res", "RMSE_q", "Rel_L2_q", "Total_Steps"
    ])

print(f"🚀 共 9 组 2W 反演实验。总成绩单将保存至: {csv_file}")

# ==========================================================
# 回调函数集合 (与cqqx代码结构完全对齐)
# ==========================================================
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
        self.steps = []
        self.vals = []

    def on_epoch_end(self):
        step = self.model.train_state.step
        if step % 500 == 0 or step <= 10:
            val = self.target_var.item()
            self.steps.append(step)
            self.vals.append(val)
            if step % 5000 == 0:
                print(f"    [Step {step:05d}] 🔮 预测 {self.var_name} = {val:.6f} | a_param = {self.a_param.item():.6f}")


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
# 2. 核心实验大循环
# ==========================================================
for nl in noise_levels:
    print(f"\n\n{'#' * 60}")
    print(f"### 🌊 2W 噪声水平: {nl * 100:.0f}% ###")
    print(f"{'#' * 60}")

    for exp_idx, a4_init in enumerate(alpha4_inits, 1):
        dde.config.set_random_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        print(f"\n{'=' * 55}")
        print(f"▶️ 实验 {exp_idx}/3 | 噪声: {nl * 100:.0f}% | 初始 a4: {a4_init:.1f}")
        print(f"{'=' * 55}")

        alpha1, alpha2, alpha3 = 0.4, 1.0, -1.0
        alpha4_true = -2.0

        out_dir = f'Data_2W_Inverse_alpha4_6Layer_Noise_{int(nl * 100)}_Init_{a4_init:.1f}'
        os.makedirs(out_dir, exist_ok=True)

        alpha4_var = dde.Variable(a4_init)


        def compute_k(alpha4_val):
            disc = alpha3 ** 2 - 4 * alpha2 * (alpha1 + 0.5 * alpha4_val)
            return (-alpha3 - np.sqrt(disc)) / (2 * alpha2)


        def exact_solution(xyt):
            x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
            k = compute_k(alpha4_true)
            a4 = alpha4_true

            # ================= 2W 阶解公式 (严格保留) =================
            G21 = (-192 * k ** 4 * y ** 4 - 1152 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 - 960 * a4 ** 4 * t ** 4
                   - 768 * k ** 3 * x * y ** 3 - 2304 * k * a4 ** 2 * t ** 2 * x * y - 1152 * k ** 2 * x ** 2 * y ** 2
                   - 1152 * a4 ** 2 * t ** 2 * x ** 2 - 768 * k * x ** 3 * y - 288 * k ** 2 * y ** 2 - 864 * a4 ** 2 * t ** 2
                   - 192 * x ** 4 - 576 * k * x * y - 288 * x ** 2 + 36)
            G22 = (-384 * k ** 4 * y ** 4 - 768 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 - 384 * a4 ** 4 * t ** 4
                   - 1536 * k ** 3 * x * y ** 3 - 1536 * k * a4 ** 2 * t ** 2 * x * y - 2304 * k ** 2 * x ** 2 * y ** 2
                   - 768 * a4 ** 2 * t ** 2 * x ** 2 - 1536 * k * x ** 3 * y + 576 * k ** 2 * y ** 2 - 192 * a4 ** 2 * t ** 2
                   - 384 * x ** 4 + 1152 * k * x * y + 576 * x ** 2 + 360)
            H2 = (
                        64 * k ** 6 * y ** 6 + 192 * k ** 4 * a4 ** 2 * t ** 2 * y ** 4 + 192 * k ** 2 * a4 ** 4 * t ** 4 * y ** 2
                        + 64 * a4 ** 6 * t ** 6 + 384 * k ** 5 * x * y ** 5 + 768 * k ** 3 * a4 ** 2 * t ** 2 * x * y ** 3
                        + 384 * k * a4 ** 4 * t ** 4 * x * y + 960 * k ** 4 * x ** 2 * y ** 4 + 1152 * k ** 2 * a4 ** 2 * t ** 2 * x ** 2 * y ** 2
                        + 192 * a4 ** 4 * t ** 4 * x ** 2 + 1280 * k ** 3 * x ** 3 * y ** 3 + 768 * k * a4 ** 2 * t ** 2 * x ** 3 * y + 48 * k ** 4 * y ** 4
                        - 288 * k ** 2 * a4 ** 2 * t ** 2 * y ** 2 + 960 * k ** 2 * x ** 4 * y ** 2 + 432 * a4 ** 4 * t ** 4 + 192 * a4 ** 2 * t ** 2 * x ** 4
                        + 192 * k ** 3 * x * y ** 3 - 576 * k * a4 ** 2 * t ** 2 * x * y + 384 * k * x ** 5 * y + 288 * k ** 2 * x ** 2 * y ** 2 - 288 * a4 ** 2 * t ** 2 * x ** 2
                        + 64 * x ** 6 + 192 * k * x ** 3 * y + 108 * k ** 2 * y ** 2 + 396 * a4 ** 2 * t ** 2 + 48 * x ** 4 + 216 * k * x * y + 108 * x ** 2 + 9)

            frac_real = G21 / H2
            frac_imag = (-a4 * G22 * t) / H2
            amp_real = 1 + frac_real
            amp_imag = frac_imag
            phase_cos = np.cos(a4 * t)
            phase_sin = np.sin(-a4 * t)

            q_real = amp_real * phase_cos - amp_imag * phase_sin
            q_imag = amp_real * phase_sin + amp_imag * phase_cos
            return np.hstack([q_real, q_imag])


        # ==========================================================
        # 核心控制方程 (标准NLS，反演 alpha4)
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

            F_R = -v_t + alpha1 * u_xx + alpha2 * u_yy + alpha3 * u_xy - alpha4_var * q_norm_sq * u
            F_I = u_t + alpha1 * v_xx + alpha2 * v_yy + alpha3 * v_xy - alpha4_var * q_norm_sq * v

            return [F_R, F_I]


        # ==========================================================
        # 2W 收缩空间域 [-5, 5]
        x_min, x_max, y_min, y_max, t_min, t_max = -5, 5, -5, 5, 0.0, 1.0
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

        # 内部随机观测点 (撒盐加噪声)
        obs_num = 1000
        obs_pts = geomtime.random_points(obs_num)
        obs_data = exact_solution(obs_pts)

        if nl > 0:
            std_u = np.std(obs_data[:, 0])
            std_v = np.std(obs_data[:, 1])
            obs_data[:, 0] += nl * std_u * np.random.randn(obs_num)
            obs_data[:, 1] += nl * std_v * np.random.randn(obs_num)

        observe_u = dde.icbc.PointSetBC(obs_pts, obs_data[:, 0:1], component=0)
        observe_v = dde.icbc.PointSetBC(obs_pts, obs_data[:, 1:2], component=1)

        data = dde.data.TimePDE(
            geomtime, pde, [ic_real, ic_imag, bc_real, bc_imag, observe_u, observe_v],
            num_domain=10000, num_boundary=2000, num_initial=2000, anchors=obs_pts, num_test=5000
        )
        data.train_distribution = "LHS"
        data.resample_train_points()

        # 🌟 对齐结构: 6个隐藏层，每层 40 宽
        a_param = torch.tensor([0.1], requires_grad=True, device=device)

        def adaptive_tanh(x):
            return torch.tanh(10.0 * a_param * x)

        net = dde.nn.FNN([3] + [40] * 6 + [2], adaptive_tanh, "Glorot normal")
        model = dde.Model(data, net)

        # ==========================================================
        # 3. 训练过程 (完全对齐两阶段迭代步数 + 动态权重)
        # ==========================================================
        loss_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0]
        train_vars = [alpha4_var, a_param]

        print(">>> Phase 1: Adam Training (10000 steps)...")
        model.compile("adam", lr=1e-3, loss_weights=loss_w, external_trainable_variables=train_vars)

        lr_cb = ExponentialLR()
        var_cb = VarLogger(alpha4_var, "alpha4", a_param)
        weight_decay_cb = LossWeightDecay(lambda_d0=100.0, lambda_d_min=10.0, tau=2000.0)
        monitor_cb = MonitorLogger()

        losshistory, train_state = model.train(
            iterations=10000, display_every=5000,
            callbacks=[var_cb, lr_cb, weight_decay_cb, monitor_cb]
        )

        print(">>> Phase 2: L-BFGS Fine-tuning (max 10000 steps)...")
        dde.optimizers.config.set_LBFGS_options(
            maxiter=10000, maxfun=10000, maxcor=50,
            ftol=1.0 * np.finfo(float).eps, gtol=1e-5, maxls=50
        )
        model.compile("L-BFGS", loss_weights=model.loss_weights, external_trainable_variables=train_vars)
        losshistory2, train_state2 = model.train(
            display_every=5000, callbacks=[var_cb, monitor_cb]
        )

        # ==========================================================
        # 4. 指标精密计算与无损存盘
        # ==========================================================
        final_a4 = alpha4_var.item()
        relative_error_a4 = abs(final_a4 - alpha4_true) / abs(alpha4_true)

        # 去重提取连续参数轨迹
        cleaned_history = list(dict.fromkeys(zip(var_cb.steps, var_cb.vals)))
        all_steps, all_vals = zip(*cleaned_history)
        all_steps, all_vals = list(all_steps), list(all_vals)

        total_iters = train_state2.step if train_state2.step > 10000 else train_state.step

        history_data = np.column_stack((all_steps, all_vals))
        np.savetxt(
            f'{out_dir}/trajectory_alpha4.txt', history_data,
            header="Iterations Recovered_alpha4", comments="", fmt='%.6f'
        )

        # 全场 100x100x21 极密网格复数场评估 (对齐 2W 域 [-5, 5])
        x_eval = np.linspace(x_min, x_max, 100)
        y_eval = np.linspace(y_min, y_max, 100)
        t_eval = np.linspace(t_min, t_max, 21)
        X, Y, T = np.meshgrid(x_eval, y_eval, t_eval)
        xyt_eval = np.hstack((X.ravel()[:, None], Y.ravel()[:, None], T.ravel()[:, None]))

        q_exact = exact_solution(xyt_eval)
        q_pred = model.predict(xyt_eval)

        u_exact, v_exact = q_exact[:, 0], q_exact[:, 1]
        u_pred, v_pred = q_pred[:, 0], q_pred[:, 1]

        err_sq = (u_pred - u_exact) ** 2 + (v_pred - v_exact) ** 2
        rmse_q = np.sqrt(np.mean(err_sq))

        norm_diff_q = np.sqrt(np.sum(err_sq))
        norm_exact_q = np.sqrt(np.sum(u_exact ** 2 + v_exact ** 2))
        rel_L2_q = norm_diff_q / norm_exact_q

        # 初始时刻 PDE 自洽残差 IC_PDE_Res (严格保留)
        x_ic = np.linspace(x_min, x_max, 80)
        y_ic = np.linspace(y_min, y_max, 80)
        X_ic, Y_ic = np.meshgrid(x_ic, y_ic)
        xyt_ic = np.hstack((X_ic.ravel()[:, None], Y_ic.ravel()[:, None], np.full(X_ic.size, t_min)[:, None]))

        xyt_tensor = torch.tensor(xyt_ic, dtype=torch.float32, requires_grad=True).to(device)
        q_tensor = model.net(xyt_tensor)

        res_list = pde(xyt_tensor, q_tensor)
        res_u_val = res_list[0].detach().cpu().numpy()
        res_v_val = res_list[1].detach().cpu().numpy()

        ic_pde_res = np.sqrt(np.mean(res_u_val ** 2 + res_v_val ** 2))

        # 💾 写入总表 (字段完全保留)
        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{int(nl * 100)}%",
                f"{a4_init:.1f}",
                f"{final_a4:.6f}",
                f"{relative_error_a4:.2e}",
                f"{ic_pde_res:.2e}",
                f"{rmse_q:.2e}",
                f"{rel_L2_q:.2e}",
                total_iters
            ])

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(
            f"✅ 实验完成！Recovered a4: {final_a4:.6f} | E_a4: {relative_error_a4:.2e} | RMSE: {rmse_q:.2e} | Rel_L2: {rel_L2_q:.2e}")

print(f"\n🎉 9 组 2W 反问题实验 (6层 + 动态权重版) 全部跑完！总成绩单详见: {csv_file}")