"""
robust_bag.py (구 predict_robust_bag.py) — 배깅 견고 블렌드 (시드+아키텍처+데이터 다양성)
챔피언(20 GRU-full + 20 ODE-full, Public 0.702)에 bootstrap 배깅 다양성을 더한다.
두 후보 생성:
  robust40 : 10 GRU-full + 10 ODE-full + 10 GRU-bag + 10 ODE-bag (사이즈 동일=40, 다양성↑)
  robust60 : 20 GRU-full + 20 ODE-full + 10 GRU-bag + 10 ODE-bag (챔피언 + 배깅, 더 견고)
실행: python -m src.inference.robust_bag → submissions/submission_{ts}_robust40.csv, _robust60.csv
"""
import sys, glob, os
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import src.inference.gru_ode_blend as pb
from src.features.clean import build_features_clean, normalize
DEV = torch.device('cpu') if os.environ.get('FORCE_CPU') else pb.DEVICE; STATS = pb.STATS
AttnGRU, ODEModel = pb.AttnGRU, pb.ODEModel

# 블렌드 구성: (prefix, 시드리스트, 모델팩토리)
# 인터림(ODE-bag 학습중 = 자정전 제출용): 챔피언과 동일 사이즈40·균형20/20, GRU절반만 배깅
ROBUST40G = [
    ('gru_full',    range(0, 10), lambda: AttnGRU()),
    ('gru_bag_full', range(0, 10), lambda: AttnGRU()),
    ('ode_full',  range(0, 20), lambda: ODEModel(nsteps=4)),
]
ROBUST40 = [
    ('gru_full',      range(0, 10), lambda: AttnGRU()),
    ('ode_full',    range(0, 10), lambda: ODEModel(nsteps=4)),
    ('gru_bag_full',   range(0, 10), lambda: AttnGRU()),
    ('ode_bag_full', range(0, 10), lambda: ODEModel(nsteps=4)),
]
ROBUST60 = [
    ('gru_full',      range(0, 20), lambda: AttnGRU()),
    ('ode_full',    range(0, 20), lambda: ODEModel(nsteps=4)),
    ('gru_bag_full',   range(0, 10), lambda: AttnGRU()),
    ('ode_bag_full', range(0, 10), lambda: ODEModel(nsteps=4)),
]


def load_test():
    paths = sorted(glob.glob(str(ROOT/'Data'/'test'/'*.csv')))
    seqs = []; scals = []; rots = []; bases = []
    for p in paths:
        X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy()
        s, sc, rot, base, _ = build_features_clean(X); sn, scn = normalize(s, sc, STATS)
        seqs.append(sn); scals.append(scn); rots.append(rot); bases.append(base)
    seq = torch.tensor(np.stack(seqs)); scal = torch.tensor(np.stack(scals))
    rot = np.stack(rots); base = np.stack(bases); mask = torch.ones(len(seq), 11)
    flip = torch.tensor([1, 4, 7, 10])
    ids = [os.path.basename(p)[:-4] for p in paths]
    return seq, scal, rot, base, mask, flip, ids


def predict_one(fp, factory, seq, scal, rot, base, mask, flip):
    m = factory().to(DEV); m.load_state_dict(torch.load(fp, map_location=DEV, weights_only=False)['model_state']); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), 256):
            s = seq[i:i+256].to(DEV); c = scal[i:i+256].to(DEV); mk = mask[i:i+256].to(DEV)
            pr = m(s, c, mk).cpu().numpy(); sf = s.clone(); sf[:, :, flip] *= -1
            pf = m(sf, c, mk).cpu().numpy(); pf[:, 1] *= -1
            out.append((pr+pf)/2)
    r = np.concatenate(out)
    return base + np.einsum('bij,bj->bi', rot.transpose(0, 2, 1), r)


def build_blend(groups, tensors):
    seq, scal, rot, base, mask, flip, _ = tensors
    preds = []; counts = {}
    for prefix, seeds, factory in groups:
        c = 0
        for k in seeds:
            fp = str(ROOT/'models'/f'{prefix}_{k}.pt')
            if not os.path.exists(fp):
                continue
            preds.append(predict_one(fp, factory, seq, scal, rot, base, mask, flip)); c += 1
        counts[prefix] = c
    return np.mean(preds, axis=0), counts, len(preds)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    jobs = {'interim': [('robust40g', ROBUST40G)],
            'full':    [('robust40', ROBUST40), ('robust60', ROBUST60)],
            'all':     [('robust40g', ROBUST40G), ('robust40', ROBUST40), ('robust60', ROBUST60)]}[mode]
    tensors = load_test(); ids = tensors[6]
    ts = datetime.now().strftime('%m%d_%H%M')
    print(f'Device: {DEV} | mode={mode} | Test: {len(ids):,}')
    for label, groups in jobs:
        ens, counts, total = build_blend(groups, tensors)
        fn = str(ROOT/'submissions'/f'submission_{ts}_{label}.csv')
        pd.DataFrame({'id': ids, 'x': ens[:, 0], 'y': ens[:, 1], 'z': ens[:, 2]}).to_csv(fn, index=False)
        print(f'  {label}: {total}모델 {counts} → {os.path.basename(fn)}', flush=True)
    print('완료')


if __name__ == '__main__':
    main()
