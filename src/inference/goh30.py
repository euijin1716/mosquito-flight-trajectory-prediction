"""
goh30.py (구 predict_blend_H.py) — GRU + ODE + HyperPhysics(H) 3-아키텍처 블렌드 ⭐ 메인
H는 위치를 직접 예측(pred_global), GRU/ODE는 잔차+base. 모두 위치 평균(등가중).
린(lean) 블렌드 지향: 기본 10 GRU + 10 ODE + 10 H = 30모델 (robust60보다 적고 다양성↑).
실행: NG=10 NO=10 NH=10 NP=0 NR=0 python -m src.inference.goh30 → submissions/submission_{ts}_GOH{N}.csv
"""
import sys, glob, os
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import src.inference.robust_bag as prb
import src.training.hyperphysics_oof as th
pp = rr = None  # P/R(실패실험)은 이 패키지에 미포함 (GOH30은 NP=NR=0) — predict_P/R 함수는 보존용
DEV = prb.DEV


def load_test_raw():
    paths = sorted(glob.glob(str(ROOT/'Data'/'test'/'*.csv')))
    X = np.stack([pd.read_csv(p)[['x', 'y', 'z']].to_numpy(np.float32) for p in paths])
    ids = [os.path.basename(p)[:-4] for p in paths]
    return torch.tensor(X, dtype=torch.float32), ids


def predict_H(fp, Xt):
    m = th.HyperPhysics_xy2().to(DEV); m.load_state_dict(torch.load(fp, map_location=DEV, weights_only=False)['model_state']); m.eval()
    def fwd(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                b = X[i:i+256].to(DEV)
                ft, df, pl, t, _, _, _, Rt, sp, _, _ = m.get_features(b, m.mean_stats, m.std_stats)
                pp, _, _ = m(ft, df, pl, t, sp, Rt)
                out.append(pp.cpu().numpy())
        return np.concatenate(out)
    pr = fwd(Xt)
    Xf = Xt.clone(); Xf[:, :, 1] *= -1            # Y-flip TTA (거울 대칭)
    pf = fwd(Xf); pf[:, 1] *= -1
    return (pr + pf) / 2


def predict_P(fp, Xt):
    m = pp.PhysTaylor().to(DEV); m.load_state_dict(torch.load(fp, map_location=DEV, weights_only=False)['model_state']); m.eval()
    def fwd(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 256):
                b = X[i:i+256].to(DEV)
                ft, df, pl, _, _, _, _, Rt, _, _, _ = m.get_features(b, m.mean_stats, m.std_stats)
                out.append(m(ft, df, pl, Rt).cpu().numpy())
        return np.concatenate(out)
    pr = fwd(Xt); Xf = Xt.clone(); Xf[:, :, 1] *= -1; pf = fwd(Xf); pf[:, 1] *= -1
    return (pr + pf) / 2


def predict_R(fp, Xt):
    # R(BankedTurn)은 Y-flip 등변성 깨짐(340샘플>1cm) → TTA 금지, raw 예측만 (OOF가 no-TTA로 검증한 형태)
    m = rr.BankedTurn().to(DEV); m.load_state_dict(torch.load(fp, map_location=DEV, weights_only=False)['model_state']); m.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), 256):
            b = Xt[i:i+256].to(DEV)
            ft, df, pl, _, _, _, _, Rt, _, _, _ = m.get_features(b, m.mean_stats, m.std_stats)
            out.append(m(ft, df, pl, Rt).cpu().numpy())
    return np.concatenate(out)


def main():
    NG = int(os.environ.get('NG', 10)); NO = int(os.environ.get('NO', 10)); NH = int(os.environ.get('NH', 10)); NP = int(os.environ.get('NP', 0)); NR = int(os.environ.get('NR', 0))
    tensors = prb.load_test(); seq, scal, rot, base, mask, flip, ids = tensors
    Xt, _ = load_test_raw()
    print(f'Device {DEV} | Test {len(ids):,} | GRU×{NG} ODE×{NO} H×{NH} P×{NP} R×{NR}')

    preds = {'GRU': [], 'ODE': [], 'H': [], 'P': [], 'R': []}
    for k in range(NG):
        fp = str(ROOT/'models'/f'gru_full_{k}.pt')
        if os.path.exists(fp): preds['GRU'].append(prb.predict_one(fp, lambda: prb.AttnGRU(), seq, scal, rot, base, mask, flip))
    for k in range(NO):
        fp = str(ROOT/'models'/f'ode_full_{k}.pt')
        if os.path.exists(fp): preds['ODE'].append(prb.predict_one(fp, lambda: prb.ODEModel(nsteps=4), seq, scal, rot, base, mask, flip))
    for k in range(NH):
        fp = str(ROOT/'models'/f'hp_full_{k}.pt')
        if os.path.exists(fp): preds['H'].append(predict_H(fp, Xt))
    for k in range(NP):
        fp = str(ROOT/'models'/f'p_full_{k}.pt')
        if os.path.exists(fp): preds['P'].append(predict_P(fp, Xt))
    for k in range(NR):
        fp = str(ROOT/'models'/f'r_full_{k}.pt')
        if os.path.exists(fp): preds['R'].append(predict_R(fp, Xt))
    for k in preds: print(f'  {k}: {len(preds[k])}개 로드')

    allp = preds['GRU'] + preds['ODE'] + preds['H'] + preds['P'] + preds['R']
    ens = np.mean(allp, 0)
    import glob as _g
    g30 = sorted(_g.glob(str(ROOT/'submissions'/'submission_*_GOH30.csv')))
    if g30:
        ref = pd.read_csv(g30[-1])[['x', 'y', 'z']].values; d = np.linalg.norm(ens - ref, axis=1)
        print(f'GOH30(0.7024) 대비 변위: 평균 {d.mean()*100:.3f}cm 최대 {d.max()*100:.3f}cm | >1cm(판정바뀔샘플) {(d>0.01).sum()}개')

    ts = datetime.now().strftime('%m%d_%H%M')
    n = len(allp); tag = 'GOHR' if NR > 0 else ('GOHP' if NP > 0 else 'GOH')
    fn = str(ROOT/'submissions'/f'submission_{ts}_{tag}{n}.csv')
    pd.DataFrame({'id': ids, 'x': ens[:, 0], 'y': ens[:, 1], 'z': ens[:, 2]}).to_csv(fn, index=False)
    print(f'\n{tag} {n}모델 블렌드 → {os.path.basename(fn)}')


if __name__ == '__main__':
    main()
