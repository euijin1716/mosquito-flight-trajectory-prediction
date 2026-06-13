"""
base.py (구 src/preprocess.py)
-------------
단일 샘플 전처리 파이프라인:
  1. CSV 로드
  2. Kalman Filter (노이즈 제거 + 위치/속도 추정)
  3. Yaw Rotation (마지막 속도 방향으로 좌표계 정렬)
  4. 시계열 피처 추출 (11, 13)   ← v2: +jerk(3) +angular_vel(1)
  5. 스칼라 피처 추출 (14,)      ← v2: +last_dir_change(1) +last_normal_accel(1) +speed_bins(5)
"""

import numpy as np
import pandas as pd

# ── 상수 ──────────────────────────────────────────────────────────────────────
DT        = 0.04   # timestep 간격 (40ms)
PRED_DT   = 0.08   # 예측 시점까지 간격 (80ms = 2 step)
CLIP_THR  = 1.33   # 센서 클리핑 기준 속력 (m/s)

# 속력 구간 경계 (EDA 3.2: 구간별 R-Hit 차이 큼)
SPEED_BINS = [0.0, 0.3, 0.6, 0.9, 1.2, np.inf]


# ── 1. CSV 로드 ───────────────────────────────────────────────────────────────
def load_sample(path) -> np.ndarray:
    """
    단일 CSV 파일 로드.
    Returns
    -------
    positions : (11, 3) float32
    """
    df = pd.read_csv(path)
    return df[['x', 'y', 'z']].to_numpy(dtype=np.float32)


# ── 2. Kalman Filter ──────────────────────────────────────────────────────────
def kalman_filter(positions: np.ndarray,
                  sigma_obs: float = 3e-4,
                  sigma_proc: float = 1e-2):
    """
    Constant-Velocity Kalman Filter.
    State  : [x, y, z, vx, vy, vz]
    Observe: [x, y, z]
    """
    n  = len(positions)
    F  = np.eye(6, dtype=np.float64)
    F[0, 3] = DT; F[1, 4] = DT; F[2, 5] = DT

    H = np.zeros((3, 6), dtype=np.float64)
    H[0, 0] = 1; H[1, 1] = 1; H[2, 2] = 1

    R = (sigma_obs  ** 2) * np.eye(3, dtype=np.float64)
    Q = (sigma_proc ** 2) * np.eye(6, dtype=np.float64)

    v0 = (positions[1] - positions[0]) / DT
    x  = np.concatenate([positions[0], v0]).astype(np.float64)
    P  = np.eye(6, dtype=np.float64)

    smoothed_pos, smoothed_vel = [], []
    for i in range(n):
        x = F @ x;  P = F @ P @ F.T + Q
        innov = positions[i].astype(np.float64) - H @ x
        S = H @ P @ H.T + R;  K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ innov;    P = (np.eye(6) - K @ H) @ P
        smoothed_pos.append(x[:3].copy())
        smoothed_vel.append(x[3:].copy())

    return (np.array(smoothed_pos, dtype=np.float32),
            np.array(smoothed_vel, dtype=np.float32))


def kalman_predict(positions: np.ndarray,
                   sigma_obs: float = 3e-4,
                   sigma_proc: float = 1e-2):
    """Kalman Filter 적용 후 +80ms 위치 예측."""
    smoothed_pos, smoothed_vel = kalman_filter(positions, sigma_obs, sigma_proc)
    kalman_pred = smoothed_pos[-1] + smoothed_vel[-1] * PRED_DT
    return kalman_pred, smoothed_pos, smoothed_vel


# ── 3. Yaw Rotation ───────────────────────────────────────────────────────────
def yaw_rotation_matrix(velocity: np.ndarray) -> np.ndarray:
    """마지막 속도의 수평 방향을 x축으로 정렬하는 회전 행렬 (z축 고정)."""
    vx, vy   = float(velocity[0]), float(velocity[1])
    speed_xy = np.sqrt(vx ** 2 + vy ** 2)
    if speed_xy < 1e-6:
        return np.eye(3, dtype=np.float32)
    cos_yaw, sin_yaw = vx / speed_xy, vy / speed_xy
    return np.array([
        [ cos_yaw, sin_yaw, 0.0],
        [-sin_yaw, cos_yaw, 0.0],
        [ 0.0,     0.0,     1.0],
    ], dtype=np.float32)


# ── 4. 시계열 피처 추출 (11, 13) ─────────────────────────────────────────────
def extract_seq_features(smoothed_pos: np.ndarray,
                         smoothed_vel: np.ndarray,
                         rot: np.ndarray) -> np.ndarray:
    """
    Yaw Rotation 후 시계열 피처 생성.

    [0:3]  relative_pos   — 마지막 관측 대비 상대 위치 (회전 후)
    [3:6]  velocity        — 속도 벡터 (회전 후)
    [6:9]  acceleration    — 가속도 벡터 (회전 후)
    [9:12] jerk            — 가속도 변화율 (EDA 2.3)
    [12]   angular_vel     — 연속 timestep 간 방향 cosine similarity (EDA 3.4)

    Returns : (11, 13) float32
    """
    last_pos = smoothed_pos[-1]

    rel_pos = (smoothed_pos - last_pos) @ rot.T   # (11, 3)
    vel_rot = smoothed_vel @ rot.T                 # (11, 3)

    # ── 가속도: 중앙 차분 ──────────────────────────────────────────────────
    accel = np.zeros_like(vel_rot)
    accel[1:-1] = (vel_rot[2:] - vel_rot[:-2]) / (2 * DT)
    accel[0]    = accel[1]
    accel[-1]   = accel[-2]

    # ── Jerk (가속도 변화율): 중앙 차분 (EDA 2.3) ─────────────────────────
    jerk = np.zeros_like(accel)
    jerk[1:-1] = (accel[2:] - accel[:-2]) / (2 * DT)
    jerk[0]    = jerk[1]
    jerk[-1]   = jerk[-2]

    # ── Angular velocity (방향 변화율): 연속 timestep 간 cosine (EDA 3.4) ──
    # v_norm: (11, 3), 각 timestep의 단위 속도 벡터
    speed = np.linalg.norm(vel_rot, axis=1, keepdims=True)   # (11, 1)
    v_norm = vel_rot / (speed + 1e-12)                        # (11, 3)
    # cos_sim[i] = dot(v_norm[i-1], v_norm[i])  →  (10,)
    cos_sim = (v_norm[:-1] * v_norm[1:]).sum(axis=1)          # (10,)
    # timestep 0은 timestep 1값 복사
    angular_vel = np.concatenate([[cos_sim[0]], cos_sim])     # (11,)

    features = np.concatenate([
        rel_pos,                    # (11, 3)
        vel_rot,                    # (11, 3)
        accel,                      # (11, 3)
        jerk,                       # (11, 3)
        angular_vel[:, None],       # (11, 1)
    ], axis=1)                      # → (11, 13)

    return features.astype(np.float32)


# ── 5. 스칼라 피처 추출 (14,) ─────────────────────────────────────────────────
def extract_scalar_features(smoothed_pos: np.ndarray,
                             smoothed_vel: np.ndarray) -> np.ndarray:
    """
    샘플 수준 스칼라 피처 (14-dim):

    [기존 7개]
      [0]  last_speed        — 마지막 속력
      [1]  last_accel        — 마지막 가속도 크기
      [2]  mean_accel        — 평균 가속도 크기
      [3]  linearity         — 궤적 선형성 (R²)
      [4]  clip_flag         — 클리핑 여부 (last_speed > 1.33)
      [5]  dir_consistency   — 연속 방향 일관성 평균
      [6]  delta_speed       — 마지막 구간 속력 변화

    [신규 7개 — EDA 근거]
      [7]  last_dir_change   — 마지막 2 timestep 간 방향 cosine (EDA 2.4)
      [8]  last_normal_accel — 마지막 법선 가속도 크기 (EDA 3.1 횡방향 오차)
      [9~13] speed_bin       — 속력 구간 one-hot 5개 (EDA 3.2)

    Returns : (14,) float32
    """
    speeds = np.linalg.norm(smoothed_vel, axis=1)   # (11,)
    last_speed = float(speeds[-1])

    vel_diff  = np.diff(smoothed_vel, axis=0) / DT  # (10, 3)
    accel_mag = np.linalg.norm(vel_diff, axis=1)    # (10,)
    last_accel = float(accel_mag[-1])
    mean_accel = float(accel_mag.mean())

    # 선형성 (R²)
    t = np.arange(len(smoothed_pos), dtype=np.float32)
    r2_list = []
    for dim in range(3):
        y = smoothed_pos[:, dim]
        coeffs = np.polyfit(t, y, 1)
        y_pred = np.polyval(coeffs, t)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2_list.append(1.0 - ss_res / (ss_tot + 1e-10))
    linearity = float(np.mean(r2_list))

    clip_flag = float(last_speed > CLIP_THR)

    v_norm = smoothed_vel / (np.linalg.norm(smoothed_vel, axis=1, keepdims=True) + 1e-12)
    cos_sim_all = (v_norm[:-1] * v_norm[1:]).sum(axis=1)
    dir_consistency = float(cos_sim_all.mean())
    delta_speed = float(speeds[-1] - speeds[-2])

    # ── 신규 ① last_dir_change: 마지막 2 timestep 방향 변화 (EDA 2.4) ──
    last_dir_change = float(cos_sim_all[-1])   # cos(angle) 마지막 구간

    # ── 신규 ② last_normal_accel: 마지막 법선 가속도 (EDA 3.1) ──────────
    # 가속도를 속도 방향(tangential)과 수직(normal)으로 분해
    last_vel_norm = v_norm[-1]                              # (3,)
    last_accel_vec = vel_diff[-1]                           # (3,) m/s²
    tangential  = np.dot(last_accel_vec, last_vel_norm) * last_vel_norm
    normal_vec  = last_accel_vec - tangential
    last_normal_accel = float(np.linalg.norm(normal_vec))

    # ── 신규 ③ speed_bin: 속력 구간 one-hot 5개 (EDA 3.2) ───────────────
    speed_bin = np.zeros(5, dtype=np.float32)
    for k in range(5):
        if SPEED_BINS[k] <= last_speed < SPEED_BINS[k + 1]:
            speed_bin[k] = 1.0
            break

    scalar = np.array([
        last_speed,
        last_accel,
        mean_accel,
        linearity,
        clip_flag,
        dir_consistency,
        delta_speed,
        last_dir_change,
        last_normal_accel,
    ], dtype=np.float32)

    return np.concatenate([scalar, speed_bin])   # (14,)


# ── 통합 파이프라인 ────────────────────────────────────────────────────────────
def preprocess_sample(path,
                      sigma_obs: float = 3e-4,
                      sigma_proc: float = 1e-2):
    """
    단일 샘플 전처리 파이프라인.

    Returns
    -------
    seq_features    : (11, 13) float32
    scalar_features : (14,)   float32
    kalman_pred     : (3,)    float32
    rot             : (3, 3)  float32
    last_pos        : (3,)    float32
    """
    positions = load_sample(path)
    kalman_pred, smoothed_pos, smoothed_vel = kalman_predict(
        positions, sigma_obs, sigma_proc
    )
    last_vel = smoothed_vel[-1]
    rot      = yaw_rotation_matrix(last_vel)

    seq_features    = extract_seq_features(smoothed_pos, smoothed_vel, rot)
    scalar_features = extract_scalar_features(smoothed_pos, smoothed_vel)

    return seq_features, scalar_features, kalman_pred, rot, positions[-1]
