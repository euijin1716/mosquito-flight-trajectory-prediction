"""
hyperphysics_oof.py (구 train_phaseH.py) — HyperPhysics_xy2 (참고코드 [LB_0.699+] 물리 추론 모델) 포팅
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
현 1등이 공유한 물리 gray-box 외삽기. pred = last_pos + R·[damp·Rodrigues(v_ema,ω) + damp·a_ema].
ω(각속도)로 곡선비행 회전 외삽 + θ/speed 게이팅(고속·급회전에서만). 우리 약점(곡선) 정조준.
우리 5-fold OOF 체계(KFold seed=42)로 래핑 → GRU/ODE/T OOF와 동일 split = 블렌드 검증 가능.
모델/피처/데이터셋은 원본 verbatim, 학습루프만 우리 fold+EMA로.

실행: python -m src.training.hyperphysics_oof --fold 0   (게이트)  |  python -m src.training.hyperphysics_oof (5-fold+OOF)
결과: models/hp_fold_{0~4}.pt, experiments/hp_oof.npz
"""
import sys, time, os, argparse, subprocess, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
EXP = ROOT/'experiments'; MODEL = ROOT/'models'
K_FOLDS = 5; SEED = 42; BATCH = 256
EPOCHS = int(os.environ.get('HEPOCHS', 16)); EMA_DECAY = 0.9; EMA_START = 6
HTAG = os.environ.get('HTAG', '')          # 레시피 변형 테스트용 (clobber 방지): hp{HTAG}_fold_*.pt
SCHED = os.environ.get('HSCHED', 'step')   # step(기본) | cosine


def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def get_device():
    return (torch.device('cuda') if torch.cuda.is_available()
            else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))


# ───────────────────────── 원본 cell[3]: SlidingWindowDataset ─────────────────────────
class SlidingWindowDataset(Dataset):
    def __init__(self, X, y, min_win=3, mode="extended", device="cpu"):
        X_tensor = torch.tensor(X, dtype=torch.float32); y_tensor = torch.tensor(y, dtype=torch.float32)
        windows = []
        for i in range(len(X)):
            targets = [4, 5, 6, 7, 8, 9, 10, 12] if mode == "extended" else [12, 10]
            for target_idx in targets:
                end_idx = target_idx - 2
                max_w = end_idx + 2 if mode == "extended" else (12 if target_idx == 12 else 10)
                for w in range(min_win, max_w):
                    windows.append((i, w, target_idx))
        X_list = []; y_list = []
        for i, w, target_idx in windows:
            X_orig = X_tensor[i]; end_idx = target_idx - 2
            pts = X_orig[end_idx - w + 1: end_idx + 1]
            target = y_tensor[i] if target_idx == 12 else X_orig[target_idx]
            if w < 11:
                v0 = pts[1] - pts[0]; n_pad = 11 - w
                js = torch.arange(n_pad, 0, -1, dtype=torch.float32)
                pad = pts[0:1] - js.unsqueeze(1) * v0.unsqueeze(0)
                X_padded = torch.cat([pad, pts], dim=0)
            else:
                X_padded = pts.clone()
            X_list.append(X_padded); y_list.append(target)
        self.X_all = torch.stack(X_list).to(device); self.y_all = torch.stack(y_list).to(device)
        diffs = self.X_all[:, 1:] - self.X_all[:, :-1]
        n1 = diffs[:, 1:].norm(dim=2).clamp(min=1e-8); n2 = diffs[:, :-1].norm(dim=2).clamp(min=1e-8)
        cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2) / (n1 * n2)).clamp(-1, 1)
        theta_last = torch.acos(cos_t[:, -1])
        self.theta_weights = (1.0 + 4.0 * (theta_last / 1.0).clamp(0, 1)).cpu().numpy()

    def __len__(self): return len(self.X_all)
    def __getitem__(self, idx): return self.X_all[idx], self.y_all[idx]


# ───────────────────────── 원본 cell[5]: 피처/손실/블록 ─────────────────────────
def _ema_va_local(diffs_local, alpha, beta):
    B, T, _ = diffs_local.shape
    one_m_a = 1.0 - alpha; one_m_b = 1.0 - beta
    vs = diffs_local.new_empty(B, T, 3); v = diffs_local[:, 0]; vs[:, 0] = v
    for t in range(1, T):
        v = alpha * diffs_local[:, t] + one_m_a * v; vs[:, t] = v
    vl = vs[:, -1]
    ad = vs[:, 1:] - vs[:, :-1]; a = ad[:, 0]
    for t in range(1, T - 1):
        a = beta * ad[:, t] + one_m_b * a
    return vl, a


def _soft_hit_loss(pred, target, thr=0.013012, k=408.348):
    return (1 - torch.sigmoid(-(torch.norm(pred - target, dim=1) - thr) * k)).mean()


def extract_features(X, mean_stats=None, std_stats=None, dir_net=None, heading_mode="3step"):
    device = X.device
    p_last = X[:, 10]; diffs = X[:, 1:] - X[:, :-1]
    n1 = diffs[:, 1:].norm(dim=2, keepdim=True) + 1e-8; n2 = diffs[:, :-1].norm(dim=2, keepdim=True) + 1e-8
    cos_t = ((diffs[:, 1:] * diffs[:, :-1]).sum(dim=2, keepdim=True) / (n1 * n2)).clamp(-1, 1)
    theta_seq = torch.acos(cos_t).squeeze(2)
    theta = theta_seq[:, -1:]; theta_mean = theta_seq.mean(1, keepdim=True); theta_std = theta_seq.std(1, keepdim=True)
    theta_vel = theta_seq[:, -1:] - theta_seq[:, -2:-1]
    theta_acc = theta_seq[:, -1:] - 2 * theta_seq[:, -2:-1] + theta_seq[:, -3:-2]
    theta_trend = theta_seq[:, -1:] - theta_seq[:, -3:].mean(1, keepdim=True)
    if dir_net is not None:
        speed_seq = diffs.norm(dim=2); state = torch.cat([speed_seq, theta_seq], dim=1)
        if dir_net[0].in_features == 29:
            z_speed_seq = diffs[:, :, 2].abs(); state = torch.cat([state, z_speed_seq], dim=1)
        weights = F.softmax(dir_net(state), dim=1); v_sm = (diffs * weights.unsqueeze(2)).sum(dim=1)
    else:
        v_sm = (3 * diffs[:, -1] + 2 * diffs[:, -2] + diffs[:, -3]) / 6.0 if heading_mode == "3step" else diffs[:, -1]
    fwd = v_sm / (v_sm.norm(dim=1, keepdim=True) + 1e-8)
    up_w = torch.zeros_like(fwd); up_w[:, 2] = 1.0
    up_w[fwd[:, 2].abs() > 0.99] = torch.tensor([0., 1., 0.], device=device)
    right = torch.cross(fwd, up_w, dim=1); right = right / (right.norm(dim=1, keepdim=True) + 1e-8)
    up = torch.cross(right, fwd, dim=1); up = up / (up.norm(dim=1, keepdim=True) + 1e-8)
    R = torch.stack([fwd, right, up], dim=2)
    v_last = diffs[:, -1]; v_prev1 = diffs[:, -2]; speed = v_last.norm(dim=1, keepdim=True)
    a_last = v_last - v_prev1; acc_mag = a_last.norm(dim=1, keepdim=True)
    v_local = torch.matmul(v_last.unsqueeze(1), R).squeeze(1)
    a_local = torch.matmul(a_last.unsqueeze(1), R).squeeze(1)
    X_local = torch.matmul(X - p_last.unsqueeze(1), R); p_std_local = X_local.std(1)
    v_local_abs = v_local.abs()
    jerk_g = diffs[:, -1] - 2 * diffs[:, -2] + diffs[:, -3]
    jerk_l = torch.matmul(jerk_g.unsqueeze(1), R).squeeze(1); jerk_mag = jerk_g.norm(dim=1, keepdim=True)
    features = torch.cat([v_local, a_local, speed, acc_mag, theta, theta_mean, theta_std, theta_trend,
                          theta_vel, theta_acc, p_std_local, v_local_abs, jerk_l, jerk_mag], dim=1)
    if mean_stats is None or std_stats is None:
        mean_stats = features.mean(0, keepdim=True); std_stats = features.std(0, keepdim=True) + 1e-8
    return (features - mean_stats) / std_stats, diffs, p_last, theta, theta_mean, theta_std, theta_seq, R, speed, mean_stats, std_stats


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)
    def forward(self, x): return self.ln(x + self.net(x))


class PriorBiasedLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_bias):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.register_buffer('prior_bias', prior_bias.clone().detach())
        with torch.no_grad():
            nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x) + self.prior_bias


def rodrigues_rotate(v, w):
    theta = w.norm(dim=1, keepdim=True); k = w / (theta + 1e-8)
    cos_t = torch.cos(theta); sin_t = torch.sin(theta)
    dot = (v * k).sum(dim=1, keepdim=True); cross = torch.cross(k, v, dim=1)
    return v * cos_t + cross * sin_t + k * dot * (1.0 - cos_t)


# ───────────────────────── 원본 cell[7]: HyperPhysics_xy2 ─────────────────────────
class HyperPhysics_xy2(nn.Module):
    def __init__(self, input_dim=24, **kwargs):
        super().__init__()
        self.sh_thr = kwargs.pop('sh_thr', 0.013012); self.sh_k = kwargs.pop('sh_k', 408.348044)
        self.mse_w = kwargs.pop('mse_w', 129.172037); self.local_w = kwargs.pop('local_w', 0.050941)
        self.theta_thr = kwargs.pop('theta_thr', float(os.environ.get('HTHETA', 1.087618))); self.speed_thr = kwargs.pop('speed_thr', float(os.environ.get('HSPEED', 0.034583)))
        self.lr = 0.005400; self.wd = 0.005659
        self.register_buffer("mean_stats", torch.zeros(1, input_dim)); self.register_buffer("std_stats", torch.ones(1, input_dim))
        prior_dir = torch.tensor([-10., -10., -10., -10., -10., -10., -10., 0., 0.693, 1.098])
        self.dir_net = nn.Sequential(nn.Linear(29, 24), nn.LayerNorm(24), nn.GELU(), PriorBiasedLinear(24, 10, prior_dir))
        prior_ema = torch.zeros(6)
        self.temporal_net = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(), PriorBiasedLinear(32, 6, prior_ema))
        prior_dyn = torch.tensor([0., 0., 0., 0., 0., 0.] + [-4.] * 24)
        self.dynamics_net = nn.Sequential(nn.Linear(input_dim, 96), nn.LayerNorm(96), nn.GELU(), ResBlock(96), PriorBiasedLinear(96, 30, prior_dyn))
        self.omega_w = nn.Parameter(torch.tensor([0.0, -0.5, -1.0]))
        self.omega_net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 48), nn.GELU(), nn.Linear(48, 3))
        with torch.no_grad():
            nn.init.normal_(self.omega_net[-1].weight, std=0.01); nn.init.zeros_(self.omega_net[-1].bias)
        self.diffusion_net = nn.Sequential(nn.Linear(input_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Linear(32, 3))

    def get_features(self, X, mean_stats=None, std_stats=None):
        return extract_features(X, mean_stats, std_stats, self.dir_net, heading_mode="3step")

    @staticmethod
    def _rotation_vector(d_prev, d_curr):
        n_prev = d_prev.norm(dim=1, keepdim=True).clamp(min=1e-8); n_curr = d_curr.norm(dim=1, keepdim=True).clamp(min=1e-8)
        d_hat_prev = d_prev / n_prev; d_hat_curr = d_curr / n_curr
        cross = torch.linalg.cross(d_hat_prev, d_hat_curr, dim=1); sin_t = cross.norm(dim=1, keepdim=True).clamp(min=1e-8)
        cos_t = (d_hat_prev * d_hat_curr).sum(1, keepdim=True).clamp(-0.9999, 0.9999); theta = torch.atan2(sin_t, cos_t)
        speed_gate = torch.sigmoid((n_prev + n_curr) * 500 - 5)
        return cross / sin_t * theta * speed_gate

    def forward(self, features, diffs, p_last, theta, speed, R):
        B = diffs.shape[0]
        ema_raw = self.temporal_net(features[:, 8:17])
        alpha = torch.sigmoid(ema_raw[:, 0:3]) * 0.8 + 0.1; beta = torch.sigmoid(ema_raw[:, 3:6]) * 0.199 + 0.8
        dyn_raw = self.dynamics_net(features)
        w_v = 2.0 + dyn_raw[:, 0:3]; w_a = 1.0 + dyn_raw[:, 3:6]
        v_local_abs = features[:, 17:20]; v_local_abs2 = v_local_abs * v_local_abs; theta2 = theta * theta
        exp_v = (F.softplus(dyn_raw[:, 6:9]) * v_local_abs + F.softplus(dyn_raw[:, 9:12]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 12:15]) * theta + F.softplus(dyn_raw[:, 15:18]) * theta2)
        exp_a = (F.softplus(dyn_raw[:, 18:21]) * v_local_abs + F.softplus(dyn_raw[:, 21:24]) * v_local_abs2 +
                 F.softplus(dyn_raw[:, 24:27]) * theta + F.softplus(dyn_raw[:, 27:30]) * theta2)
        diffs_local = torch.matmul(diffs, R)
        vl, al = _ema_va_local(diffs_local, alpha, beta)
        diff_speed = diffs_local.norm(dim=2)
        def rv_masked(ka, kb):
            rv = self._rotation_vector(diffs_local[:, ka], diffs_local[:, kb])
            valid = ((diff_speed[:, ka] > 1e-5) & (diff_speed[:, kb] > 1e-5)).float()
            return rv * valid.unsqueeze(1), valid
        ov1, vm1 = rv_masked(-2, -1); ov2, vm2 = rv_masked(-3, -2); ov3, vm3 = rv_masked(-4, -3)
        w_logits = self.omega_w.view(1, 3).expand(B, -1)
        masks = torch.stack([vm1, vm2, vm3], dim=1)
        w_attn = F.softmax(w_logits.masked_fill(masks == 0, -1e9), dim=1)
        omega_hist = (w_attn[:, 0].unsqueeze(1) * ov1 + w_attn[:, 1].unsqueeze(1) * ov2 + w_attn[:, 2].unsqueeze(1) * ov3)
        current_speed = speed.view(B, 1) if speed is not None else diff_speed[:, -1].unsqueeze(1)
        omega_speed_gate = torch.sigmoid(current_speed * 500 - 5)
        omega_delta = self.omega_net(features) * omega_speed_gate
        theta_scalar = theta.view(B, 1)
        theta_gate = torch.sigmoid((theta_scalar - self.theta_thr) * 10)
        speed_gate_strong = torch.sigmoid((current_speed - self.speed_thr) * 200)
        rotation_gate = theta_gate * speed_gate_strong
        omega = (omega_hist + omega_delta) * rotation_gate
        v_rotated = rodrigues_rotate(vl, omega)
        pred_local = (w_v * torch.exp(-exp_v)) * v_rotated + (w_a * torch.exp(-exp_a)) * al
        log_var = self.diffusion_net(features).clamp(min=-5.0, max=5.0)
        pred_global = p_last + torch.einsum('nij,nj->ni', R, pred_local)
        return pred_global, pred_local, log_var

    def compute_loss(self, pp, yr, pred_local=None, yr_local=None, log_var=None, **kwargs):
        sh = _soft_hit_loss(pp, yr, thr=self.sh_thr, k=self.sh_k)
        loss = sh + self.mse_w * F.mse_loss(pp, yr)
        if pred_local is not None and yr_local is not None and log_var is not None:
            squared_error = (pred_local - yr_local) ** 2
            nll_loss = 0.5 * (torch.exp(-log_var) * squared_error + log_var)
            loss = loss + self.local_w * nll_loss.mean()
        return loss


# ───────────────────────── 우리 fold OOF 래퍼 ─────────────────────────
def load_train():
    paths = sorted((ROOT/'Data'/'train').glob('*.csv'))
    labs = pd.read_csv(ROOT/'Data'/'train_labels.csv').sort_values('id').reset_index(drop=True)[['x', 'y', 'z']].to_numpy(np.float32)
    X = np.stack([pd.read_csv(p)[['x', 'y', 'z']].to_numpy(np.float32) for p in paths])
    return X, labs


def r_hit(p, t, thr=0.01): return float(np.mean(np.linalg.norm(p - t, axis=1) <= thr))


def run_fold(fold):
    dev = get_device(); set_seed(SEED + fold)
    X, Y = load_train()
    splits = list(KFold(K_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(Y))))
    tr_traj, va_traj = splits[fold]
    train_ds = SlidingWindowDataset(X[tr_traj], Y[tr_traj], min_win=3, mode="extended", device=dev)
    sampler = WeightedRandomSampler(train_ds.theta_weights, len(train_ds), replacement=True)
    loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
    Xva = torch.tensor(X[va_traj], dtype=torch.float32, device=dev); Yva = torch.tensor(Y[va_traj], dtype=torch.float32, device=dev)
    model = HyperPhysics_xy2().to(dev)
    with torch.no_grad():
        *_, mn, st = model.get_features(torch.tensor(X[tr_traj], dtype=torch.float32, device=dev))
        model.mean_stats.copy_(mn); model.std_stats.copy_(st)
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)
    sch = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS) if SCHED == 'cosine'
           else torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.6))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def eval_state(state):
        raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(state); model.eval()
        with torch.no_grad():
            ft, df, pl, th, _, _, _, Rt, sp, _, _ = model.get_features(Xva, model.mean_stats, model.std_stats)
            pp, _, _ = model(ft, df, pl, th, sp, Rt)
            rh = (torch.norm(pp - Yva, dim=1) <= 0.01).float().mean().item()
        model.load_state_dict(raw); return rh, pp.cpu().numpy()

    best = 0.0; bep = 0; path = MODEL/f'hp{HTAG}_fold_{fold}.pt'; t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train(); tot = 0.0; n = 0
        for Xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            ft, df, pl, th, _, _, _, Rt, sp, _, _ = model.get_features(Xb, model.mean_stats, model.std_stats)
            pp, pred_local, log_var = model(ft, df, pl, th, sp, Rt)
            yr_local = torch.matmul((yb - pl).unsqueeze(1), Rt).squeeze(1)
            loss = model.compute_loss(pp, yb, pred_local, yr_local, log_var)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(Xb); n += len(Xb)
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1 - EMA_DECAY)
                else: ema[k] = v.detach().clone()
        vr, _ = eval_state(model.state_dict())
        vr_e, _ = eval_state(ema) if ep >= EMA_START else (0.0, None)
        use_e = vr_e > vr; vbest = max(vr, vr_e)
        if vbest > best:
            best, bep = vbest, ep
            torch.save({'fold': fold, 'epoch': ep, 'model_state': (ema if use_e else model.state_dict()),
                        'val_rhit': vbest, 'ema': bool(use_e)}, path)
        print(f'  fold{fold} ep{ep:2d} loss={tot/n:.4f} raw={vr:.4f} ema={vr_e:.4f}{" ★"+("E" if use_e else "") if vbest==best else ""} ({(time.time()-t0)/60:.1f}min)', flush=True)
    print(f'fold{fold} 완료 best R-Hit={best:.4f} (ep{bep}, {(time.time()-t0)/60:.1f}min)', flush=True)


def merge():
    dev = get_device(); X, Y = load_train()
    splits = list(KFold(K_FOLDS, shuffle=True, random_state=SEED).split(np.arange(len(Y))))
    oof = np.zeros((len(Y), 3), np.float32); frh = []
    for fold, (_, va_traj) in enumerate(splits):
        ck = torch.load(MODEL/f'hp{HTAG}_fold_{fold}.pt', map_location=dev, weights_only=False); frh.append(ck['val_rhit'])
        model = HyperPhysics_xy2().to(dev); model.load_state_dict(ck['model_state']); model.eval()
        Xva = torch.tensor(X[va_traj], dtype=torch.float32, device=dev)
        with torch.no_grad():
            ft, df, pl, th, _, _, _, Rt, sp, _, _ = model.get_features(Xva, model.mean_stats, model.std_stats)
            pp, _, _ = model(ft, df, pl, th, sp, Rt)
        oof[va_traj] = pp.cpu().numpy()
    orh = r_hit(oof, Y)
    print('=' * 60); print(f'Fold R-Hit: {[f"{r:.4f}" for r in frh]}')
    print(f'HyperPhysics OOF R-Hit = {orh:.4f}  (GRU 0.6766/ODE 0.6777 대비 {orh-0.6766:+.4f}/{orh-0.6777:+.4f})')
    np.savez(EXP/f'hp{HTAG}_oof.npz', preds=oof, true=Y)
    print(f'저장: hp{HTAG}_oof.npz')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fold', type=int, default=None); ap.add_argument('--merge', action='store_true'); a = ap.parse_args()
    if a.fold is not None: run_fold(a.fold); return
    if a.merge: merge(); return
    print(f'HyperPhysics 5-fold | EPOCHS={EPOCHS} | device={get_device()}')
    t0 = time.time()
    def sub(f): return f, subprocess.run([sys.executable, str(Path(__file__).resolve()), '--fold', str(f)]).returncode
    with ThreadPoolExecutor(max_workers=K_FOLDS) as ex:
        for fut in as_completed({ex.submit(sub, k): k for k in range(K_FOLDS)}):
            f, c = fut.result(); print(f'  fold {f} {"완료" if c==0 else "오류"} ({(time.time()-t0)/60:.1f}min)')
    merge(); print(f'전체 {(time.time()-t0)/60:.1f}분')


if __name__ == '__main__':
    main()
