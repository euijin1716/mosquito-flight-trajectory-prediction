# DACON 모기 비행 궤적 예측 — 우승 솔루션 (Private LB 0.7035)

- **대회 링크**: https://dacon.io/competitions/official/236716/overview/description
- **코드 공유 링크**: https://dacon.io/competitions/official/236716/codeshare/14013?page=1&dtype=recent


11스텝(−400~0ms, 40ms 간격) 3D 위치 시퀀스로 **+80ms 위치**를 예측한다.
평가 지표는 **R-Hit@1cm** = 예측이 정답으로부터 유클리드 1cm 이내인 비율.

> 📥 **데이터:** 대회 데이터(`Data/`)는 재배포 금지라 repo에 포함하지 않는다. DACON에서 받아 `Data/{train,test}/*.csv`, `Data/train_labels.csv` 로 배치하면 된다.
> 데이터 링크: https://dacon.io/competitions/official/236716/data

| 구분 | 점수 | 순위 |
|---|---|---|
| Public LB | 0.7024 | 7등 |
| **Private LB (최종)** | **0.7035** | **1등** |

최종 제출물 = **GOH30**: GRU + Neural ODE + HyperPhysics 3개 아키텍처를 각 10시드씩, 등가중 평균한 30모델 블렌드.

---

## 1. 핵심 접근

**공통 레시피 (3개 멤버 모두 적용):**
- **Base = 등속 외삽(cv_1step)** : `pred_base = last + 2·(last − prev)` (+80ms = 2스텝)
- **Yaw 회전 정렬 프레임** : 마지막 속도를 x축에 정렬 → 회전 등변성
- **잔차 학습** : 모델은 `(정답 − base)`를 회전 프레임에서 예측
- **Soft R-Hit 손실** : 1cm 경계에 그라디언트 집중 (mode 최적화)
- **내부전이 사전학습** : 궤적 내부 지점에서 "+2스텝" 예측을 추가 학습 → 데이터 증폭
- **EMA 가중치 + Y-flip TTA**

**3개 멤버 (서로 메커니즘적으로 탈상관 → 블렌드 다양성):**
| 멤버 | 메커니즘 | 코드 |
|---|---|---|
| **GRU** | 양방향 GRU + attention pooling | `src/training/gru_*.py` |
| **Neural ODE** | 댐핑 가속도장을 RK4 적분 (GRU 인코더 latent) | `src/training/ode_*.py` |
| **HyperPhysics** | 물리 gray-box: roll 기반 Rodrigues 회전 외삽 + 속도/각도 게이팅 | `src/training/hyperphysics_*.py` |

> 💡 **HyperPhysics가 천장 돌파의 핵심.** 모기 선회는 yaw가 아니라 **roll(뱅킹)로 양력벡터를 기울여** 만든다는 물리를 모델링 → GRU·ODE와 탈상관된 신호를 잡아 0.702 천장을 넘어 1등 달성.

**블렌드:** 30모델(10+10+10)의 예측 **위치**를 등가중 평균. GRU/ODE/H 각각 Y-flip TTA 적용.

---

## 2. 디렉터리 구조

```
mosquito-flight-trajectory-prediction/
├── README.md
├── requirements.txt               # 고정 버전 의존성 (재현 기준 환경)
├── Data/                          # 원본 데이터 (입력)
│   ├── train/*.csv                #   학습 궤적 10,000개 (각 11스텝 x,y,z)
│   ├── train_labels.csv           #   +80ms 정답 좌표
│   └── test/*.csv                 #   테스트 궤적 10,000개
├── experiments/                   # 전처리 산출물 (아티팩트)
│   ├── norm_stats.npz            #   피처 정규화 통계 (seq/scalar 평균·표준편차)
│   └── train_cache.npz           #   GRU/ODE 학습 캐시 (gitignore · build_cache 생성, 50k 예시)
├── models/                        # 학습된 가중치 (.pt · gitignore — 재학습으로 생성)
│   ├── gru_full_0~9.pt            #   GRU 10시드
│   ├── ode_full_0~9.pt            #   ODE 10시드
│   └── hp_full_0~9.pt             #   HyperPhysics 10시드
├── submissions/                   # 최종 산출물 (제출 csv)
│   └── submission_0531_1800_GOH30.csv   # 🏆 최종 제출 = Private 0.7035
└── src/                           # 전체 코드 패키지
    ├── features/                  # 피처 추출·정규화 (모든 학습/추론이 공유)
    │   ├── base.py                #   yaw회전·seq/scalar 피처·load_sample (구 preprocess.py)
    │   └── clean.py               #   build_features_clean, normalize (구 features_clean.py)
    ├── training/                  # 학습 (가중치 생성)
    │   ├── gru_oof.py             #   GRU 5-fold OOF + 캐시빌드 (구 train_phaseG.py)
    │   ├── gru_full.py            #   GRU 전체데이터 학습 (구 train_phaseG_full.py)
    │   ├── ode_oof.py  ode_full.py            # Neural ODE (구 train_phaseODE*.py)
    │   ├── hyperphysics_oof.py                # HyperPhysics (구 train_phaseH.py)
    │   └── hyperphysics_full.py               # (구 train_phaseH_full.py)
    ├── inference/                 # 추론·블렌드
    │   ├── goh30.py               #   ⭐ 메인: GOH30 블렌드 생성 (구 predict_blend_H.py)
    │   ├── gru_ode_blend.py       #   AttnGRU·ODEModel 정의 + norm_stats 로드 (구 predict_blend_ode.py)
    │   └── robust_bag.py          #   load_test, predict_one 공용 추론 유틸 (구 predict_robust_bag.py)
    └── tools/
        ├── build_cache.py         #   train_cache.npz 빌드 (학습 전 1회, ~15초)
        └── build_norm_stats.py    #   norm_stats.npz 재생성 (동봉본 유실 시)
```

> ✅ **경로 독립 실행.** 모든 스크립트가 자기 위치 기준 절대경로(`ROOT`)를 쓰므로 어느 디렉터리에서 실행해도 된다. 권장 형태는 프로젝트 루트에서 `python -m src.inference.goh30` (모듈 실행); `python src/inference/goh30.py` (파일 실행)도 동일하게 동작한다.

---

## 3. 환경

- **Python 3.11** (개발 환경: conda `DA_project`)
- 패키지 (고정 버전 = 재현 기준 환경):
  ```bash
  pip install -r requirements.txt
  ```
- 디바이스: CUDA / Apple MPS / CPU 자동 감지 (코드가 알아서 선택)

---

## 4. 재현 방법

> 명령은 프로젝트 루트(`mosquito-flight-trajectory-prediction/`)에서 `-m` 모듈 형태 기준. (스크립트가 경로 독립적이라 `python src/inference/goh30.py`처럼 파일 경로 실행도 어디서든 동작.)

> DACON에서 받은 데이터를 `Data/`에 배치한 뒤 진행한다. (학습 가중치 `.pt`는 repo 미포함 → 재학습으로 생성)

```bash
cd mosquito-flight-trajectory-prediction

# 1) 학습 캐시 빌드 (필수 — git 미포함, 난수 없이 ~15초)
python -m src.tools.build_cache

# 2) 모델 학습 (각 10시드, 0~9)
for s in 0 1 2 3 4 5 6 7 8 9; do
  PREFIX=gru_full RW=2.0 RT=0.0015 python -m src.training.gru_full   --seed $s
  PREFIX=ode_full RW=2.0 RT=0.0015 python -m src.training.ode_full   --seed $s
  PREFIX=hp_full                   python -m src.training.hyperphysics_full --seed $s
done
#   → models/gru_full_*.pt, ode_full_*.pt, hp_full_*.pt

# 3) 블렌드 → submissions/submission_<날짜>_GOH30.csv
NG=10 NO=10 NH=10 NP=0 NR=0 python -m src.inference.goh30
```

> **환경변수 의미** — `RW`=Soft R-Hit 손실 가중치(2.0), `RT`=손실 τ(0.0015) : GRU/ODE의 우승 레시피값(생략 시 기본 0.5/0.003이라 **반드시 지정**). HyperPhysics는 자체 손실이라 불필요. `NG/NO/NH`=블렌드에 넣을 GRU/ODE/H 모델 수, `NP/NR`=0(실패 실험, 미사용).

---

## 5. 재현성

전 과정을 패키지 안에서 실제로 돌려 감사했다 (raw → 전처리 → 학습 → 최종).

| 단계 | 비트 단위 재현 | 검증 방법 |
|---|---|---|
| 전처리 `train_cache.npz` | ✅ **비트동일** | raw에서 `build_cache` 재생성 → 9개 키 `array_equal=True` |
| 전처리 `norm_stats.npz` | ✅ 동봉본 한정 | 동봉본은 비트동일 · raw 재계산 시 ~1e-6 (float 노이즈, 무시 가능) |
| **모델 학습 (`.pt`)** | ✅ **비트동일** | 시드 고정 + 이 환경(PyTorch 2.12 / MPS)에서 연산 재현 가능 |
| **최종 CSV (raw부터 재학습)** | ✅ **비트동일** | 30개 전부 재학습 → 블렌드 = GOH30, 변위 **0.0000cm** |

**결론:** 이 환경에서는 **raw 데이터부터 30개를 재학습해도 0.7035가 비트 단위로 완벽 재현**된다 (실측 변위 0.0000cm). 모든 난수가 시드 고정(`torch.manual_seed`·`np.random.seed`)이고 학습 연산이 매번 같은 결과를 내기 때문이다.

> ⚠️ **환경 의존성:** 위 비트동일은 **동일 환경**(같은 하드웨어·PyTorch·MPS) 기준. 다른 머신/PyTorch 버전/CPU↔GPU 전환 시에는 float 연산 순서 차이로 비트동일이 깨질 수 있으나, **점수(~0.7035)는 재현**된다.

> 📦 **최소 입력 정리:**
> - **raw 데이터(`Data/`) + 모든 `.py` + `norm_stats.npz`(1.3KB)** → 처음부터 **비트 단위** 완벽 재현 (동일 환경).
> - **`norm_stats.npz`가 없어도 동작** — 학습/추론 스크립트가 raw에서 **자동 생성**한다(`src/features/norm_stats.py`의 `load_or_build`). 명시적 재생성은 `python -m src.tools.build_norm_stats`. 단 ~1e-6 차이 → **점수 재현**(비트동일은 동봉본 필요).
>   - 즉 **npz 있으면 로드(비트) / 없으면 raw 자동 생성(점수)** 의 두 모드를 모두 지원.
> - `train_cache.npz`·30개 `.pt`는 **git 미포함** — raw + .py로 재생성(`build_cache` ~15초 / 학습 ~50분).

---

## 6. 파일별 역할

**추론 — `src/inference/` (GOH30 생성):**
- `goh30.py` — ⭐ 메인. GRU+ODE+H 30모델 로드 → Y-flip TTA → 등가중 평균 → CSV
- `robust_bag.py` — `load_test`(테스트 피처 빌드), `predict_one`(단일모델 추론+TTA)
- `gru_ode_blend.py` — `AttnGRU`·`ODEModel` 클래스 정의, `norm_stats` 로드
- (H 클래스는 `src/training/hyperphysics_oof.py`의 `HyperPhysics_xy2`를 import)

**학습 — `src/training/` (가중치 생성):**
- `gru_oof.py` / `gru_full.py` — GRU 모델·캐시빌드(5-fold OOF) / 전체데이터 EMA 학습
- `ode_oof.py` / `ode_full.py` — Neural ODE
- `hyperphysics_oof.py` / `hyperphysics_full.py` — HyperPhysics

**유틸 — `src/tools/`:** `build_cache.py`(train_cache.npz 빌드 — 학습 전 1회) · `build_norm_stats.py`(norm_stats.npz 재생성, 유실 시 fallback)

**공용 — `src/features/`:** `base.py`, `clean.py` — 피처 추출·정규화 (모든 학습/추론이 공유)

---

## 7. 한 줄 요약

> **모기 선회의 물리(roll 뱅킹)를 모델링한 HyperPhysics를 GRU·ODE와 블렌드 → 0.702 천장 돌파 → Private 0.7035.**
> raw 데이터부터 재학습하면 우승 결과(0.7035)가 재현된다 (동일 환경에선 비트 단위).
