"""
build_norm_stats.py — raw 데이터(Data/train)에서 norm_stats.npz 재생성 (비상용 유틸).
보통은 동봉본을 쓰며 유실 시에만 사용. 학습/추론 스크립트는 npz가 없으면
src.features.norm_stats.load_or_build 로 **자동 생성**하므로, 이 CLI는 명시적 재생성용이다.
※ float 합산순서 차이로 동봉본과 ~1e-6 상이 → 점수는 동일 재현(비트동일은 동봉 .npz 사용).
실행: python -m src.tools.build_norm_stats   (→ experiments/norm_stats.npz 생성)
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.features.norm_stats import compute


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT/'experiments'/'norm_stats.npz'))
    a = ap.parse_args()
    stats = compute()
    np.savez(a.out, **stats)
    print(f'norm_stats 생성: {a.out}  (seq 13 + scalar 22 평균/표준편차)')


if __name__ == '__main__':
    main()
