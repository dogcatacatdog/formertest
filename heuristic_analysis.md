# Heuristic.py 회귀분석 기법 및 파라미터 분석

## 개요
`heuristic.py`는 제조업 장비 데이터에 대해 **Bottom-up 세그멘테이션**과 **Lasso 회귀분석**을 결합한 고급 시계열 분석 시스템입니다.

## 주요 회귀분석 기법

### 1. Lasso 회귀분석 (L1 정규화)
```python
from sklearn.linear_model import Lasso
model = Lasso(alpha=alpha).fit(df_seg[['x']], df_seg['y'])
```

**특징:**
- L1 정규화를 통한 특성 선택 효과
- 노이즈에 강한 회귀계수 추정
- 과적합 방지

### 2. OLS 회귀분석 (선형성 검증)
```python
import statsmodels.api as sm
X_const = sm.add_constant(df_seg[['x']])
ols = sm.OLS(df_seg['y'], X_const).fit()
```

**목적:**
- p-value 기반 통계적 유의성 검증
- R² 값을 통한 설명력 평가
- 선형성 vs 비선형성 분류

## 핵심 파라미터 분석

### 1. 고정 임계값 파라미터

| 파라미터 | 값 | 설명 | 영향 |
|----------|----|----- |------|
| `MSE_THRESHOLD` | 0.1 | 정규화된 MSE 임계값 | **낮을수록**: 더 엄격한 세그먼트 병합 |
| `SLOPE_DIFF_THRESHOLD` | 0.25 | 인접 세그먼트 기울기 차이 허용값 | **낮을수록**: 더 유사한 기울기만 병합 |
| `WINDOW_SIZE` | 30 | 초기 세그먼트 크기 | **클수록**: 더 긴 구간 단위로 분석 |

### 2. 동적 파라미터

#### 2.1 Heuristic Alpha (Lasso 정규화 강도)
```python
def heuristic_alpha(X, y):
    y_std = np.std(y)                    # 종속변수 표준편차
    x_range = X['x'].max() - X['x'].min()  # 독립변수 범위
    scale = y_std / x_range              # 스케일링 비율
    return max(min(scale, 1.0), 0.01)    # 0.01 ~ 1.0 범위로 제한
```

**특징:**
- 데이터의 분산과 범위에 따라 자동 조정
- 안정적인 회귀계수 추정을 위한 적응형 정규화

#### 2.2 기울기 분류 기준
```python
if abs(slope) <= 0.05:      # ±0.05 이하
    slope_sign = 'stable'
elif slope > 0:
    slope_sign = 'increase'
else:
    slope_sign = 'decrease'
```

#### 2.3 선형성 판정 기준
```python
linearity = 'linear' if (ols.pvalues[1] <= 0.05 and ols.rsquared >= 0.3) else 'non-linear'
```

**조건:**
- p-value ≤ 0.05 (95% 신뢰수준에서 유의)
- R² ≥ 0.3 (30% 이상 설명력)

### 3. 고급 병합 조건

#### 3.1 기본 병합 조건
```python
return (mse_norm < MSE_THRESHOLD) and 
       (abs(m1 - m2) < SLOPE_DIFF_THRESHOLD) and 
       (slope_var < 0.05)
```

#### 3.2 곡률 검증 (추가 조건)
```python
if np.std(df_combined['y'] - preds) < 1.0:
    curvature = np.std(np.diff(preds, n=2))  # 2차 차분의 표준편차
    if curvature > 0.05:
        return False  # 곡률이 높으면 병합 거부
```

## 알고리즘 워크플로우

### 1. 초기 세그멘테이션
```
데이터 길이: N
세그먼트 수: ⌊(N - WINDOW_SIZE) / WINDOW_SIZE⌋ + 1
각 세그먼트: 30개 데이터 포인트
```

### 2. Bottom-up 병합
```
for each adjacent pair:
    if evaluate_merge_condition():
        merge segments
    else:
        keep separate
```

### 3. 세그먼트 라벨링
```
for each final segment:
    - Lasso 회귀분석
    - OLS 통계 검증
    - 기울기 분류
    - 선형성 판정
```

## 성능 특성

### 장점
1. **적응형 정규화**: 데이터 특성에 따른 자동 alpha 조정
2. **통계적 검증**: p-value와 R²를 통한 엄격한 선형성 판정
3. **노이즈 강건성**: Lasso의 L1 정규화로 이상치에 강함
4. **다층 검증**: MSE, 기울기 차이, 곡률 등 다중 조건 검증

### 한계
1. **고정 임계값**: 도메인별 최적화 필요
2. **선형 가정**: 복잡한 비선형 패턴 감지 제한
3. **윈도우 크기**: 데이터 길이에 따른 적응형 조정 부재

## 실제 활용 시나리오

### 제조업 장비 모니터링
- **온도 센서**: 급격한 온도 변화 구간 감지
- **압력 센서**: 안정 구간 vs 변동 구간 분류
- **전력 소모**: 효율성 변화 패턴 분석

### 이상치 탐지 연계
```python
# 기울기 급변 구간 = 잠재적 이상치 영역
if abs(slope_current - slope_previous) > SLOPE_DIFF_THRESHOLD:
    flag_as_potential_anomaly()
```

## 파라미터 튜닝 가이드

### 1. MSE_THRESHOLD 조정
- **증가 (0.1 → 0.2)**: 더 관대한 병합, 더 긴 세그먼트
- **감소 (0.1 → 0.05)**: 더 엄격한 병합, 더 세밀한 분할

### 2. WINDOW_SIZE 조정
- **데이터 길이 < 100**: WINDOW_SIZE = 10~15
- **데이터 길이 100~500**: WINDOW_SIZE = 20~30 (현재 설정)
- **데이터 길이 > 500**: WINDOW_SIZE = 50~100

### 3. 도메인별 특화 설정

#### 고주파 노이즈 환경 (예: 진동 센서)
```python
MSE_THRESHOLD = 0.15        # 더 관대한 오차 허용
SLOPE_DIFF_THRESHOLD = 0.3  # 더 큰 기울기 변화 허용
```

#### 안정적 환경 (예: 온도 센서)
```python
MSE_THRESHOLD = 0.05        # 더 엄격한 오차 기준
SLOPE_DIFF_THRESHOLD = 0.1  # 더 작은 기울기 변화만 허용
```

## 결론

`heuristic.py`의 회귀분석 시스템은 **통계적 엄밀성**과 **실용적 유연성**을 균형있게 제공합니다. 특히 제조업 환경의 다양한 센서 데이터에 대해 자동화된 패턴 인식과 이상치 탐지 기능을 제공하여, 예측 유지보수와 품질 관리에 효과적으로 활용할 수 있습니다.
