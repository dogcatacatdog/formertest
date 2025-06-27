import pandas as pd
import numpy as np
import torch
from hybrid import InformerAutoformerHybrid, DynamicThreshold
from sklearn.preprocessing import StandardScaler

def simple_analysis():
    """간단한 제조업 데이터 분석"""
    
    # 원본 데이터 로드
    df = pd.read_csv('sample_manufacturing.csv')
    print("=== 제조업 데이터 분석 ===")
    print(f"총 데이터 포인트: {len(df)}")
    print(f"시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"데이터 컬럼: {list(df.columns)}")
    
    # 기본 통계
    print("\n=== 주요 센서 통계 ===")
    for col in ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']:
        mean_val = df[col].mean()
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        print(f"{col:12}: 평균={mean_val:7.2f}, 표준편차={std_val:6.2f}, 범위=[{min_val:7.2f}, {max_val:7.2f}]")
    
    # 명백한 이상치 탐지 (임계값 기반)
    print("\n=== 명백한 이상치 탐지 ===")
    temp_anomalies = df[df['TEMP'] > 450]
    power_anomalies = df[df['RF_POWER'] > 300]
    
    print(f"온도 이상치 (>450): {len(temp_anomalies)}개")
    print(f"RF 파워 이상치 (>300): {len(power_anomalies)}개")
    
    if len(temp_anomalies) > 0:
        print("\n온도 이상치 상세:")
        for idx, row in temp_anomalies.iterrows():
            print(f"  {row['timestamp']}: TEMP={row['TEMP']:.2f}, RF_POWER={row['RF_POWER']:.2f}")
    
    # 레시피별 분석
    print("\n=== 레시피별 분석 ===")
    recipe_stats = df.groupby('recipe').agg({
        'TEMP': ['mean', 'std'],
        'PRESSURE': ['mean', 'std'], 
        'RF_POWER': ['mean', 'std'],
        'CHAMBER_LOAD': ['mean', 'std']
    }).round(2)
    
    for recipe in df['recipe'].unique():
        subset = df[df['recipe'] == recipe]
        print(f"{recipe}: {len(subset)}개 샘플")
        print(f"  TEMP: {subset['TEMP'].mean():.1f}±{subset['TEMP'].std():.1f}")
        print(f"  PRESSURE: {subset['PRESSURE'].mean():.1f}±{subset['PRESSURE'].std():.1f}")
        print(f"  RF_POWER: {subset['RF_POWER'].mean():.1f}±{subset['RF_POWER'].std():.1f}")
    
    # 시간대별 패턴
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    print("\n=== 시간대별 패턴 ===")
    hourly_temp = df.groupby('hour')['TEMP'].mean()
    print(f"최고 온도 시간대: {hourly_temp.idxmax()}시 ({hourly_temp.max():.1f}°C)")
    print(f"최저 온도 시간대: {hourly_temp.idxmin()}시 ({hourly_temp.min():.1f}°C)")

def test_model_components():
    """모델 구성요소 개별 테스트"""
    print("\n=== 하이브리드 모델 구성요소 테스트 ===")
    
    # 샘플 데이터 생성
    batch_size, seq_len, feature_dim = 4, 96, 3
    x = torch.randn(batch_size, seq_len, feature_dim)
    
    # 1. AdaptiveSeriesDecomposition 테스트
    from hybrid import AdaptiveSeriesDecomposition
    decomp = AdaptiveSeriesDecomposition()
    seasonal, trend = decomp(x)
    print(f"시계열 분해: 입력={x.shape}, 계절성={seasonal.shape}, 트렌드={trend.shape}")
    
    # 2. DynamicProbSparseAttention 테스트  
    from hybrid import DynamicProbSparseAttention
    attention = DynamicProbSparseAttention(d_model=64, n_heads=4)
    x_emb = torch.randn(batch_size, seq_len, 64)
    attn_out, attn_map = attention(x_emb)
    print(f"어텐션: 입력={x_emb.shape}, 출력={attn_out.shape}")
    
    # 3. 전체 모델 테스트
    model = InformerAutoformerHybrid(
        input_dim=feature_dim,
        d_model=64,
        out_len=24,
        use_patch=True,
        patch_len=12
    )
    
    y_true = torch.randn(batch_size, 24)
    output = model(x, y_true)
    
    print(f"전체 모델:")
    print(f"  예측: {output['forecast'].shape}")
    print(f"  이상점수: {output['anomaly_score'].shape if output['anomaly_score'] is not None else 'None'}")
    print(f"  계절성: {output['seasonal'].shape}")
    print(f"  트렌드: {output['trend'].shape}")
    
    # 4. DynamicThreshold 테스트
    threshold = DynamicThreshold()
    scores = torch.randn(100) * 0.5 + 1.0  # 평균 1.0, 표준편차 0.5
    scores[95:] = torch.randn(5) * 0.2 + 2.0  # 마지막 5개를 이상치로
    
    thres_val = threshold.update(scores)
    anomalies = (scores > thres_val).sum().item()
    print(f"동적 임계값: {thres_val:.4f}, 탐지된 이상치: {anomalies}개")

if __name__ == "__main__":
    simple_analysis()
    test_model_components()
