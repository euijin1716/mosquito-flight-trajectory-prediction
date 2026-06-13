"""
gru_ode_blend.py (구 predict_blend_ode.py) — GRU + Neural ODE 구조 다양성 블렌드 (각 10시드, 총 20모델 + Y-flip TTA)
gru_full(BiGRU+attn) 0~9 + ode_full(Neural ODE) 0~9. OOF 블렌드 0.6785(역대 최고).
실행: python -m src.inference.gru_ode_blend → submissions/submission_{ts}_blendODE.csv
"""
import sys
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features.clean import build_features_clean, normalize
from src.features.norm_stats import load_or_build as _load_norm_stats

TEST_DIR = ROOT/'Data'/'test'; MODEL = ROOT/'models'
STATS = _load_norm_stats(ROOT/'experiments'/'norm_stats.npz')
Y_FLIP = [1, 4, 7, 10]; N_EACH = 10
DEVICE = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))


class AttnGRU(nn.Module):   # gru_full
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=3, dr=0.15):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.attn = nn.Linear(h*2, 1)
        self.head = nn.Sequential(nn.Linear(h*6+scal_dim, 256), nn.GELU(), nn.Dropout(dr),
                                  nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 3))
    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); last = out[:, -1, :]; m = mask.unsqueeze(-1)
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)
        score = self.attn(out).squeeze(-1).masked_fill(mask < 0.5, -1e9)
        att = (torch.softmax(score, dim=1).unsqueeze(-1)*out).sum(1)
        return self.head(torch.cat([last, mean, att, scal], -1))


class ODEModel(nn.Module):   # ode_full (ode_oof의 MaskedBiGRU와 동일)
    def __init__(self, seq_dim=13, scal_dim=22, h=128, nl=2, dr=0.15, latent=96, nsteps=4):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(seq_dim, h), nn.LayerNorm(h))
        self.gru = nn.GRU(h, h, nl, batch_first=True, bidirectional=True, dropout=dr if nl > 1 else 0)
        self.to_latent = nn.Sequential(nn.Linear(h*4+scal_dim, latent), nn.LayerNorm(latent), nn.GELU())
        self.accel = nn.Sequential(nn.Linear(3+3+latent, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dr),
                                   nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 3))
        self.damping = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        self.bias = nn.Parameter(torch.zeros(3))
        self.nsteps = nsteps; self.dt = 0.08/nsteps
    def _deriv(self, rpos, rvel, lat):
        a = self.accel(torch.cat([rpos, rvel, lat], -1))
        return rvel, -self.damping*rvel+a
    def forward(self, seq, scal, mask):
        x = self.proj(seq); out, _ = self.gru(x); m = mask.unsqueeze(-1)
        mean = (out*m).sum(1)/m.sum(1).clamp(min=1)
        lat = self.to_latent(torch.cat([out[:, -1, :], mean, scal], -1))
        rpos = torch.zeros(seq.size(0), 3, device=seq.device); rvel = torch.zeros_like(rpos)
        for _ in range(self.nsteps):
            dt = self.dt
            dp1, dv1 = self._deriv(rpos, rvel, lat)
            dp2, dv2 = self._deriv(rpos+0.5*dt*dp1, rvel+0.5*dt*dv1, lat)
            dp3, dv3 = self._deriv(rpos+0.5*dt*dp2, rvel+0.5*dt*dv2, lat)
            dp4, dv4 = self._deriv(rpos+dt*dp3, rvel+dt*dv3, lat)
            rpos = rpos+(dt/6)*(dp1+2*dp2+2*dp3+dp4)
            rvel = rvel+(dt/6)*(dv1+2*dv2+2*dv3+dv4)
        return rpos+self.bias


def main():
    paths = sorted(TEST_DIR.glob('*.csv'))
    print(f'Device: {DEVICE} | Test: {len(paths):,}')
    seqs, scals, rots, bases = [], [], [], []
    for p in tqdm(paths, desc='Test 전처리'):
        X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy()
        seq, sc22, rot, base, _ = build_features_clean(X)
        seq_n, sc_n = normalize(seq, sc22, STATS)
        seqs.append(seq_n); scals.append(sc_n); rots.append(rot); bases.append(base)
    seq = torch.tensor(np.stack(seqs)); scal = torch.tensor(np.stack(scals))
    rot = np.stack(rots); base = np.stack(bases); mask = torch.ones(len(seq), 11)
    flip = torch.tensor(Y_FLIP, dtype=torch.long)

    jobs = [(f'gru_full_{k}.pt', AttnGRU) for k in range(N_EACH)] + \
           [(f'ode_full_{k}.pt', ODEModel) for k in range(N_EACH)]
    preds = []
    for fname, cls in jobs:
        ck = torch.load(MODEL/fname, map_location=DEVICE, weights_only=False)
        model = cls().to(DEVICE); model.load_state_dict(ck['model_state']); model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(seq), 256):
                s = seq[i:i+256].to(DEVICE); c = scal[i:i+256].to(DEVICE); mk = mask[i:i+256].to(DEVICE)
                pr = model(s, c, mk).cpu().numpy()
                sf = s.clone(); sf[:, :, flip] *= -1
                pf = model(sf, c, mk).cpu().numpy(); pf[:, 1] *= -1
                out.append((pr+pf)/2.0)
        resid = np.concatenate(out)
        preds.append(base + np.einsum('bij,bj->bi', rot.transpose(0, 2, 1), resid))
        print(f'  {fname} ({cls.__name__}) 완료')

    ens = np.mean(preds, axis=0)
    sub = pd.DataFrame({'id': [p.stem for p in paths], 'x': ens[:, 0], 'y': ens[:, 1], 'z': ens[:, 2]})
    out_path = ROOT/'submissions'/f'submission_{datetime.now():%m%d_%H%M}_blendODE.csv'
    sub.to_csv(out_path, index=False)
    print(f'\n앙상블 {len(preds)}모델(GRU{N_EACH}+ODE{N_EACH}) → 저장: {out_path}\n{sub.head()}')


if __name__ == '__main__':
    main()
