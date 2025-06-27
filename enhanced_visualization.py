"""
제조업 데이터 시각화 및 이상 탐지 시스템
- 강화된 시각화 기능
- 인터랙티브 대시보드
- 장비/챔버/레시피 선택 기능
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from hybrid import InformerAutoformerHybrid, DynamicThreshold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 선택적 import (없어도 기본 기능 동작)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly가 설치되지 않았습니다. 기본 matplotlib을 사용합니다.")

def load_and_preprocess_data(csv_path, features=['TEMP', 'PRESSURE', 'RF_POWER'], target='CHAMBER_LOAD'):
    """제조업 데이터를 로드하고 전처리"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 피처 선택
    feature_data = df[features].values
    target_data = df[target].values
    
    # 정규화
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    feature_scaled = feature_scaler.fit_transform(feature_data)
    target_scaled = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
    
    return feature_scaled, target_scaled, feature_scaler, target_scaler, df

def create_sequences(data, target, seq_len=96, pred_len=24):
    """시계열 시퀀스 생성"""
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(target[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def create_interactive_plots(df, equipment_id='All', chamber_id='All', recipe='All'):
    """Plotly 기반 인터랙티브 플롯 생성"""
    if not PLOTLY_AVAILABLE:
        print("Plotly를 사용할 수 없습니다. 기본 matplotlib 플롯을 사용합니다.")
        return create_basic_plots(df, equipment_id, chamber_id, recipe)
    
    # 데이터 필터링
    filtered_df = df.copy()
    
    if equipment_id != 'All':
        filtered_df = filtered_df[filtered_df['eqp_id'] == equipment_id]
    if chamber_id != 'All':
        filtered_df = filtered_df[filtered_df['chamber_id'] == chamber_id]
    if recipe != 'All':
        filtered_df = filtered_df[filtered_df['recipe'] == recipe]
    
    if len(filtered_df) == 0:
        print("선택된 조건에 해당하는 데이터가 없습니다.")
        return
    
    # 서브플롯 생성
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            '온도 시계열', '압력 시계열',
            'RF 파워 시계열', '챔버 로드 시계열',
            '온도 vs RF 파워', '이상치 탐지'
        )
    )
    
    # 1. 온도 시계열
    fig.add_trace(
        go.Scatter(x=filtered_df['timestamp'], y=filtered_df['TEMP'],
                  mode='lines', name='온도', line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. 압력 시계열
    fig.add_trace(
        go.Scatter(x=filtered_df['timestamp'], y=filtered_df['PRESSURE'],
                  mode='lines', name='압력', line=dict(color='blue')),
        row=1, col=2
    )
    
    # 3. RF 파워 시계열
    fig.add_trace(
        go.Scatter(x=filtered_df['timestamp'], y=filtered_df['RF_POWER'],
                  mode='lines', name='RF 파워', line=dict(color='green')),
        row=2, col=1
    )
    
    # 4. 챔버 로드 시계열
    fig.add_trace(
        go.Scatter(x=filtered_df['timestamp'], y=filtered_df['CHAMBER_LOAD'],
                  mode='lines', name='챔버 로드', line=dict(color='purple')),
        row=2, col=2
    )
    
    # 5. 온도 vs RF 파워 (레시피별)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, recipe_type in enumerate(filtered_df['recipe'].unique()):
        recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
        fig.add_trace(
            go.Scatter(x=recipe_data['TEMP'], y=recipe_data['RF_POWER'],
                      mode='markers', name=f'{recipe_type}',
                      marker=dict(color=colors[i % len(colors)], size=4, opacity=0.6)),
            row=3, col=1
        )
    
    # 6. 이상치 탐지
    temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
    normal_data = filtered_df[filtered_df['TEMP'] <= 450]
    
    # 정상 데이터
    fig.add_trace(
        go.Scatter(x=normal_data['timestamp'], y=normal_data['TEMP'],
                  mode='markers', name='정상', 
                  marker=dict(color='blue', size=3, opacity=0.5)),
        row=3, col=2
    )
    
    # 이상치
    if len(temp_anomalies) > 0:
        fig.add_trace(
            go.Scatter(x=temp_anomalies['timestamp'], y=temp_anomalies['TEMP'],
                      mode='markers', name='온도 이상',
                      marker=dict(color='red', size=6, symbol='x')),
            row=3, col=2
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        height=900,
        title_text=f"제조업 데이터 분석 - 장비: {equipment_id}, 챔버: {chamber_id}, 레시피: {recipe}",
        showlegend=True
    )
    
    # 통계 출력
    print(f"\n=== 필터링된 데이터 통계 ===")
    print(f"데이터 포인트: {len(filtered_df):,}")
    print(f"온도 범위: {filtered_df['TEMP'].min():.1f} - {filtered_df['TEMP'].max():.1f}°C")
    print(f"압력 범위: {filtered_df['PRESSURE'].min():.2f} - {filtered_df['PRESSURE'].max():.2f}")
    print(f"RF 파워 범위: {filtered_df['RF_POWER'].min():.1f} - {filtered_df['RF_POWER'].max():.1f}")
    print(f"온도 이상치: {len(temp_anomalies)}개")
    
    fig.show()
    return fig

def create_basic_plots(df, equipment_id='All', chamber_id='All', recipe='All'):
    """기본 matplotlib 플롯 생성"""
    # 데이터 필터링
    filtered_df = df.copy()
    
    if equipment_id != 'All':
        filtered_df = filtered_df[filtered_df['eqp_id'] == equipment_id]
    if chamber_id != 'All':
        filtered_df = filtered_df[filtered_df['chamber_id'] == chamber_id]
    if recipe != 'All':
        filtered_df = filtered_df[filtered_df['recipe'] == recipe]
    
    if len(filtered_df) == 0:
        print("선택된 조건에 해당하는 데이터가 없습니다.")
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'제조업 데이터 분석 - 장비: {equipment_id}, 챔버: {chamber_id}, 레시피: {recipe}')
    
    # 1. 온도 시계열
    axes[0,0].plot(filtered_df['timestamp'], filtered_df['TEMP'], 'r-', alpha=0.7)
    axes[0,0].set_title('온도 시계열')
    axes[0,0].set_ylabel('온도 (°C)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True)
    
    # 2. 압력 시계열
    axes[0,1].plot(filtered_df['timestamp'], filtered_df['PRESSURE'], 'b-', alpha=0.7)
    axes[0,1].set_title('압력 시계열')
    axes[0,1].set_ylabel('압력')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True)
    
    # 3. RF 파워 시계열
    axes[1,0].plot(filtered_df['timestamp'], filtered_df['RF_POWER'], 'g-', alpha=0.7)
    axes[1,0].set_title('RF 파워 시계열')
    axes[1,0].set_ylabel('RF 파워')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True)
    
    # 4. 챔버 로드 시계열
    axes[1,1].plot(filtered_df['timestamp'], filtered_df['CHAMBER_LOAD'], 'purple', alpha=0.7)
    axes[1,1].set_title('챔버 로드 시계열')
    axes[1,1].set_ylabel('챔버 로드')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True)
    
    # 5. 온도 vs RF 파워 (레시피별)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, recipe_type in enumerate(filtered_df['recipe'].unique()):
        recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
        axes[2,0].scatter(recipe_data['TEMP'], recipe_data['RF_POWER'], 
                         c=colors[i % len(colors)], alpha=0.6, s=10, label=recipe_type)
    axes[2,0].set_title('온도 vs RF 파워 (레시피별)')
    axes[2,0].set_xlabel('온도 (°C)')
    axes[2,0].set_ylabel('RF 파워')
    axes[2,0].legend()
    axes[2,0].grid(True)
    
    # 6. 이상치 탐지
    temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
    normal_data = filtered_df[filtered_df['TEMP'] <= 450]
    
    axes[2,1].scatter(normal_data['timestamp'], normal_data['TEMP'], 
                     c='blue', alpha=0.5, s=10, label='정상')
    if len(temp_anomalies) > 0:
        axes[2,1].scatter(temp_anomalies['timestamp'], temp_anomalies['TEMP'], 
                         c='red', s=30, marker='x', label='온도 이상')
    axes[2,1].set_title('이상치 탐지')
    axes[2,1].set_xlabel('시간')
    axes[2,1].set_ylabel('온도 (°C)')
    axes[2,1].legend()
    axes[2,1].tick_params(axis='x', rotation=45)
    axes[2,1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 통계 출력
    print(f"\n=== 필터링된 데이터 통계 ===")
    print(f"데이터 포인트: {len(filtered_df):,}")
    print(f"온도 범위: {filtered_df['TEMP'].min():.1f} - {filtered_df['TEMP'].max():.1f}°C")
    print(f"압력 범위: {filtered_df['PRESSURE'].min():.2f} - {filtered_df['PRESSURE'].max():.2f}")
    print(f"RF 파워 범위: {filtered_df['RF_POWER'].min():.1f} - {filtered_df['RF_POWER'].max():.1f}")
    print(f"온도 이상치: {len(temp_anomalies)}개")

def interactive_data_explorer(csv_path='sample_manufacturing.csv'):
    """인터랙티브 데이터 탐색기"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    equipment_ids = ['All'] + sorted(df['eqp_id'].unique().tolist())
    chamber_ids = ['All'] + sorted(df['chamber_id'].unique().tolist())
    recipes = ['All'] + sorted(df['recipe'].unique().tolist())
    
    print("\n=== 인터랙티브 데이터 탐색기 ===")
    print("사용 가능한 옵션:")
    print(f"장비 ID: {equipment_ids}")
    print(f"챔버 ID: {chamber_ids}")
    print(f"레시피: {recipes}")
    
    while True:
        print(f"\n현재 선택 가능한 옵션:")
        print(f"1. 장비 ID 선택")
        print(f"2. 챔버 ID 선택")
        print(f"3. 레시피 선택")
        print(f"4. 선택된 조건으로 시각화")
        print(f"5. 종료")
        
        try:
            choice = input("선택하세요 (1-5): ").strip()
            
            if choice == '1':
                print(f"장비 ID 옵션: {equipment_ids}")
                equipment_id = input("장비 ID를 입력하세요: ").strip() or 'All'
            elif choice == '2':
                print(f"챔버 ID 옵션: {chamber_ids}")
                chamber_id = input("챔버 ID를 입력하세요: ").strip() or 'All'
            elif choice == '3':
                print(f"레시피 옵션: {recipes}")
                recipe = input("레시피를 입력하세요: ").strip() or 'All'
            elif choice == '4':
                equipment_id = getattr(interactive_data_explorer, 'equipment_id', 'All')
                chamber_id = getattr(interactive_data_explorer, 'chamber_id', 'All')
                recipe = getattr(interactive_data_explorer, 'recipe', 'All')
                
                print(f"시각화 생성 중... (장비: {equipment_id}, 챔버: {chamber_id}, 레시피: {recipe})")
                
                if PLOTLY_AVAILABLE:
                    create_interactive_plots(df, equipment_id, chamber_id, recipe)
                else:
                    create_basic_plots(df, equipment_id, chamber_id, recipe)
            elif choice == '5':
                break
            else:
                print("잘못된 선택입니다. 1-5 중에서 선택하세요.")
                
            # 선택값 저장
            if choice == '1':
                interactive_data_explorer.equipment_id = equipment_id
            elif choice == '2':
                interactive_data_explorer.chamber_id = chamber_id
            elif choice == '3':
                interactive_data_explorer.recipe = recipe
                
        except KeyboardInterrupt:
            print("\n프로그램을 종료합니다.")
            break
        except Exception as e:
            print(f"오류가 발생했습니다: {e}")

def detect_anomalies_with_model(csv_path='sample_manufacturing.csv'):
    """하이브리드 모델을 사용한 이상 탐지"""
    
    print("데이터 로드 중...")
    features, targets, feat_scaler, targ_scaler, df = load_and_preprocess_data(csv_path)
    
    seq_len = 96
    pred_len = 24
    X, y = create_sequences(features, targets, seq_len, pred_len)
    
    print(f"시퀀스 생성 완료: X shape={X.shape}, y shape={y.shape}")
    
    # 학습/테스트 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 모델 초기화
    model = InformerAutoformerHybrid(
        input_dim=3,
        d_model=64,
        n_heads=4,
        d_ff=128,
        num_layers=2,
        out_len=pred_len,
        use_patch=True,
        patch_len=12,
        share_params=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    # 모델 학습
    print("모델 학습 중...")
    model.train()
    train_losses = []
    
    for epoch in range(5):  # 빠른 테스트를 위해 에포크 수 감소
        epoch_loss = 0
        for i in range(0, len(X_train), 8):
            batch_x = X_train[i:i+8]
            batch_y = y_train[i:i+8]
            
            optimizer.zero_grad()
            output = model(batch_x, batch_y)
            loss = criterion(output['forecast'], batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / (len(X_train) // 8)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.6f}")
    
    # 이상 탐지
    print("이상 탐지 수행 중...")
    model.eval()
    thres = DynamicThreshold()
    
    all_anomaly_scores = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), 8):
            batch_x = X_test[i:i+8]
            batch_y = y_test[i:i+8]
            
            output = model(batch_x, batch_y)
            
            if output['anomaly_score'] is not None:
                all_anomaly_scores.extend(output['anomaly_score'].cpu().numpy())
    
    all_anomaly_scores = np.array(all_anomaly_scores)
    threshold = thres.update(torch.tensor(all_anomaly_scores))
    anomaly_mask = all_anomaly_scores > threshold
    
    # 결과 출력
    print(f"\n=== 이상 탐지 결과 ===")
    print(f"총 테스트 샘플: {len(all_anomaly_scores)}")
    print(f"이상 샘플 수: {anomaly_mask.sum()}")
    print(f"이상률: {anomaly_mask.mean():.2%}")
    print(f"동적 임계값: {threshold:.6f}")
    print(f"평균 이상 점수: {all_anomaly_scores.mean():.6f}")
    print(f"최대 이상 점수: {all_anomaly_scores.max():.6f}")
    
    return model, all_anomaly_scores, anomaly_mask, threshold, df

if __name__ == "__main__":
    print("=== 제조업 데이터 분석 시스템 ===")
    print("1. 인터랙티브 데이터 탐색")
    print("2. 모델 기반 이상 탐지")
    print("3. 웹 대시보드 실행")
    
    choice = input("선택하세요 (1-3) [기본값: 1]: ").strip() or "1"
    
    if choice == "1":
        interactive_data_explorer()
    elif choice == "2":
        model, scores, mask, threshold, df = detect_anomalies_with_model()
        
        anomaly_indices = np.where(mask)[0]
        print(f"\n이상으로 탐지된 시점들 (처음 5개):")
        for idx in anomaly_indices[:5]:
            print(f"  Index {idx}: Score {scores[idx]:.6f}")
    elif choice == "3":
        print("웹 대시보드를 시작하려면 다음 명령을 실행하세요:")
        print("python web_dashboard.py")
    else:
        print("잘못된 선택입니다.")
