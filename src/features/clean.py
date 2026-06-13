"""
clean.py (구 src/features_clean.py) — raw 기반 클린 전처리 (GOH30 공용 피처)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
깨진 Kalman 대신 raw 기반:
  - base   = cv_1step = last + 2*(last - prev)   (+80ms 등속, R-Hit 0.5787)
  - vel    = np.gradient(X, DT)                  (central diff)
  - rot    = yaw_rotation_matrix(vel[-1])
  - seq    = extract_seq_features(X, vel, rot)   (11,13)
  - scalar = extract_scalar_features(X, vel)(14) + 신규 8 = (22,)

train_cache.npz / norm_stats.npz 와 동일 산식.
학습(캐시 빌드)·추론 양쪽에서 이 함수를 써서 일관성 보장.
"""
import numpy as np
from .base import (
    load_sample, yaw_rotation_matrix,
    extract_seq_features, extract_scalar_features,
)

DT = 0.04


def build_features_clean(X: np.ndarray):
    """
    Parameters
    ----------
    X : (11, 3) raw positions (float)

    Returns
    -------
    seq      : (11, 13) float32   (정규화 전)
    scalar22 : (22,)    float32   (정규화 전)
    rot      : (3, 3)   float32
    base     : (3,)     float32   cv_1step +80ms 예측
    last_pos : (3,)     float32
    """
    X = X.astype(np.float64)
    vel = np.gradient(X, DT, axis=0)                 # (11,3) central diff
    rot = yaw_rotation_matrix(vel[-1])

    seq    = extract_seq_features(X, vel, rot)       # (11,13)
    base14 = extract_scalar_features(X, vel)         # (14,)

    speeds = np.linalg.norm(vel, axis=1)
    steps  = np.linalg.norm(np.diff(X, axis=0), axis=1)
    max_speed = float(speeds.max())
    speed_std = float(speeds.std())
    ms3 = float(speeds[-3:].mean())
    ms5 = float(speeds[-5:].mean())
    path_len = float(steps.sum())
    net = float(np.linalg.norm(X[-1] - X[0]))
    straight = net / (path_len + 1e-8)
    t = np.arange(11.0)
    noise = float(np.mean([
        (X[:, d] - np.polyval(np.polyfit(t, X[:, d], 2), t)).std() for d in range(3)
    ]))
    tt = np.arange(4.0)
    acc_trend = float(np.polyfit(tt, speeds[-4:], 1)[0])

    scalar22 = np.concatenate([
        base14,
        [max_speed, speed_std, ms3, ms5, path_len, straight, noise, acc_trend],
    ]).astype(np.float32)

    base = (X[-1] + 2.0 * (X[-1] - X[-2])).astype(np.float32)
    return seq.astype(np.float32), scalar22, rot.astype(np.float32), base, X[-1].astype(np.float32)


def normalize(seq, scalar, stats):
    """norm_stats.npz 통계로 정규화."""
    seq_n = ((seq - stats['seq_mean']) / stats['seq_std']).astype(np.float32)
    scal_n = ((scalar - stats['scalar_mean']) / stats['scalar_std']).astype(np.float32)
    return seq_n, scal_n
