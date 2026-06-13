"""
hyperphysics_full.py (구 train_phaseH_full.py) — HyperPhysics 전체데이터 학습 (홀드아웃 없음, 제출용, EMA)
hyperphysics_oof(구 train_phaseH)의 모델/피처/데이터셋 재사용, 전체 trajectory로 학습. θ-오버샘플링 유지.
실행: PREFIX=hp_full python -m src.training.hyperphysics_full --seed 0
결과: models/{PREFIX}_{seed}.pt (EMA 가중치)
"""
import sys, time, os, argparse
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, WeightedRandomSampler
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.training import hyperphysics_oof as tc

MODEL = ROOT/'models'
PREFIX = os.environ.get('PREFIX', 'hp_full')
FULL_EPOCHS = int(os.environ.get('HEPOCHS', 12)); EMA_DECAY = 0.9; BATCH = 256


def train_seed(seed):
    dev = tc.get_device(); tc.set_seed(1000 + seed)
    X, Y = tc.load_train(); N = len(X)
    if os.environ.get('BAG'):
        idx = np.random.RandomState(7000 + seed).choice(N, N, replace=True); X, Y = X[idx], Y[idx]
    ds = tc.SlidingWindowDataset(X, Y, min_win=3, mode="extended", device=dev)
    sampler = WeightedRandomSampler(ds.theta_weights, len(ds), replacement=True)
    loader = DataLoader(ds, batch_size=BATCH, sampler=sampler)
    model = tc.HyperPhysics_xy2().to(dev)
    with torch.no_grad():
        *_, mn, st = model.get_features(torch.tensor(X, dtype=torch.float32, device=dev))
        model.mean_stats.copy_(mn); model.std_stats.copy_(st)
    opt = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=model.wd)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.6)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for ep in range(1, FULL_EPOCHS + 1):
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
        if ep % 4 == 0 or ep == FULL_EPOCHS:
            print(f'  seed{seed} ep{ep:2d} loss={tot/n:.4f} ({(time.time()-t0)/60:.1f}min)', flush=True)
    torch.save({'seed': seed, 'epochs': FULL_EPOCHS, 'model_state': ema, 'val_rhit': -1, 'ema': True},
               MODEL/f'{PREFIX}_{seed}.pt')
    print(f'seed{seed} 완료 EMA저장 ({(time.time()-t0)/60:.1f}min)', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True); a = ap.parse_args()
    train_seed(a.seed)


if __name__ == '__main__':
    main()
