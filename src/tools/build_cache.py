"""
build_cache.py — train_cache.npz (GRU/ODE 학습 캐시, 50,000 examples) 빌드.
gru_full/ode_full 학습이 이 캐시를 요구한다. git 미포함이라 학습 전 1회 실행.
난수 없음 → 비트동일 재생성, ~15초.
실행: python -m src.tools.build_cache   (→ experiments/train_cache.npz)
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from src.training import gru_oof


def main():
    gru_oof.build_cache()


if __name__ == '__main__':
    main()
