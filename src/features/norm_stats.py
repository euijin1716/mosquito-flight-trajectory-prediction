"""
norm_stats.py — 피처 정규화 통계(seq 13 + scalar 22)의 로드/생성 (단일 소스).

듀얼 모드:
  • 동봉 norm_stats.npz 가 있으면 → 그대로 로드 (대회 당시 통계 = 비트 단위 재현)
  • 없으면 → raw 학습 데이터(Data/train)에서 즉석 생성 (점수 재현, 비트는 ~1e-6 차이)

모든 학습/추론이 load_or_build 를 거쳐 통계를 얻으므로, npz 유무와 무관하게 동작이 보장된다.
"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from .clean import build_features_clean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = ROOT / 'experiments' / 'norm_stats.npz'
TRAIN_DIR = ROOT / 'Data' / 'train'


def compute(train_dir=TRAIN_DIR):
    """raw 학습 CSV 전체에서 피처 평균/표준편차 계산 → dict(seq_mean/seq_std/scalar_mean/scalar_std)."""
    SEQ, SC = [], []
    for p in sorted(glob.glob(str(Path(train_dir) / '*.csv'))):
        X = pd.read_csv(p)[['x', 'y', 'z']].to_numpy()
        seq, sc, *_ = build_features_clean(X)
        SEQ.append(seq); SC.append(sc)
    SEQ = np.stack(SEQ); SC = np.stack(SC)
    return {
        'seq_mean': SEQ.reshape(-1, 13).mean(0), 'seq_std': SEQ.reshape(-1, 13).std(0),
        'scalar_mean': SC.mean(0),               'scalar_std': SC.std(0),
    }


def load_or_build(path=DEFAULT_PATH, train_dir=TRAIN_DIR, save=True):
    """npz가 있으면 로드(비트 재현), 없으면 raw에서 생성. save=True면 생성분을 path에 저장."""
    path = Path(path)
    if path.exists():
        z = np.load(path)
        print(f'[norm_stats] 동봉본 로드: {path.name} (비트 재현)')
        return {k: z[k] for k in z.files}
    print(f'[norm_stats] {path.name} 없음 → raw에서 생성 (점수 재현, 비트 ~1e-6 차이)…')
    stats = compute(train_dir)
    if save:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **stats)
        print(f'[norm_stats] 생성·저장: {path}')
    return stats
