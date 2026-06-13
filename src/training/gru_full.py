"""
gru_full.py (구 train_phaseG_full.py) — GRU 전체 데이터 학습 (홀드아웃 없음, 제출용)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
attention pooling + EMA 가중치 GRU를 전체 10,000개(=50,000 examples) 캐시로 재학습.
- 홀드아웃 없음 → EMA 가중치를 최종 채택 (일반화 booster)
- --seed 로 시드별 1개 학습 → goh30에서 10시드 앙상블 + Y-flip TTA

실행: PREFIX=gru_full RW=2.0 RT=0.0015 python -m src.training.gru_full --seed 0
결과: models/gru_full_{0~4}.pt  (EMA 가중치)
"""
import sys, time, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.training import gru_oof as tc

MODEL = ROOT/'models'
import os as _os
PREFIX=_os.environ.get('PREFIX','gru_full')
FULL_EPOCHS = 55
N_SEEDS = 5
EMA_DECAY = 0.9
EMA_START = 8


def train_seed(seed):
    dev = tc.get_device()
    d = np.load(tc.CACHE)
    seq = torch.tensor(d['seq']); scal = torch.tensor(d['scal']); msk = torch.tensor(d['mask']); tgt = torch.tensor(d['tgt'])
    N = len(seq)
    if _os.environ.get('BAG'): idx_all = np.random.RandomState(7000+seed).choice(N, N, replace=True)  # bootstrap
    else: idx_all = np.arange(N)
    torch.manual_seed(1000 + seed); np.random.seed(1000 + seed)
    model = tc.MaskedBiGRU().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.LR, weight_decay=tc.WD)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FULL_EPOCHS)
    flip = torch.tensor(tc.Y_FLIP, device=dev)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t0 = time.time()
    for ep in range(1, FULL_EPOCHS + 1):
        model.train(); np.random.shuffle(idx_all); tot = 0.0
        for i in range(0, N, tc.BATCH):
            b = idx_all[i:i+tc.BATCH]
            s = seq[b].to(dev); c = scal[b].to(dev); mk = msk[b].to(dev); tg = tgt[b].to(dev)
            if torch.rand(1).item() < tc.FLIP_PROB:
                s = s.clone(); s[:, :, flip] *= -1; tg = tg.clone(); tg[:, 1] *= -1
            s = s + torch.randn_like(s) * tc.NOISE_STD * mk.unsqueeze(-1)
            opt.zero_grad(); loss = tc.combined_loss(model(s, c, mk), tg); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step(); tot += loss.item()*len(b)
        sch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point: ema[k].mul_(EMA_DECAY).add_(v, alpha=1-EMA_DECAY)
                else: ema[k] = v.detach().clone()
        if ep % 10 == 0 or ep == FULL_EPOCHS:
            print(f'  seed{seed} ep{ep:3d} loss={tot/N:.4f} ({(time.time()-t0)/60:.1f}min)', flush=True)
    # EMA 가중치 채택
    torch.save({'seed': seed, 'epochs': FULL_EPOCHS, 'model_state': ema, 'val_rhit': -1, 'ema': True},
               MODEL/f'{PREFIX}_{seed}.pt')
    print(f'seed{seed} 완료 EMA저장 ({(time.time()-t0)/60:.1f}min)', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, default=None); a = ap.parse_args()
    if a.seed is not None:
        train_seed(a.seed); return
    assert tc.CACHE.exists(), f'캐시 없음 {tc.CACHE}'
    print(f'GRU 전체학습 | {FULL_EPOCHS}ep × {N_SEEDS}seeds (EMA) | device={tc.get_device()}')
    t0 = time.time()
    def sub(s): return s, subprocess.run([sys.executable, str(Path(__file__).resolve()), '--seed', str(s)]).returncode
    with ThreadPoolExecutor(max_workers=N_SEEDS) as ex:
        for fut in as_completed({ex.submit(sub, s): s for s in range(N_SEEDS)}):
            s, code = fut.result(); print(f'  seed {s} {"완료" if code==0 else "오류"} ({(time.time()-t0)/60:.1f}min)')
    print(f'전체 {(time.time()-t0)/60:.1f}분 | 모델 gru_full_0~{N_SEEDS-1}.pt')


if __name__ == '__main__':
    main()
