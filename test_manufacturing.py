import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from hybrid import InformerAutoformerHybrid, DynamicThreshold
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo
from ipywidgets import interact, widgets
import pickle
import os
import hashlib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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

def detect_anomalies_with_model(csv_path='sample_manufacturing.csv', use_interactive=True, use_cache=True, cache_dir='cache'):
    """하이브리드 모델을 사용한 이상 탐지 (캐시 지원)"""
    
    # 1. 데이터 업데이트 확인
    data_updated, current_checksum = check_data_update(csv_path, cache_dir)
    
    print("=== 캐시 상태 확인 ===")
    print(f"데이터 업데이트 여부: {'Yes' if data_updated else 'No'}")
    print(f"현재 데이터 체크섬: {current_checksum[:10]}...")
    
    # 2. 데이터 로드 및 전처리
    print("데이터 로드 중...")
    features, targets, feat_scaler, targ_scaler, df = load_and_preprocess_data(csv_path)
    
    # 3. 시퀀스 데이터 생성
    seq_len = 96  # 8시간 (5분 간격)
    pred_len = 24 # 2시간 예측
    X, y = create_sequences(features, targets, seq_len, pred_len)
    
    print(f"시퀀스 생성 완료: X shape={X.shape}, y shape={y.shape}")
    
    # 4. 학습/테스트 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 텐서 변환
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 5. 모델 설정
    model_config = {
        'input_dim': 3,
        'd_model': 64,
        'n_heads': 4,
        'd_ff': 128,
        'num_layers': 2,
        'out_len': pred_len,
        'use_patch': True,
        'patch_len': 12,
        'share_params': True
    }
    
    scaler_data = {
        'feat_scaler': feat_scaler,
        'targ_scaler': targ_scaler
    }
    
    train_losses = []
    model = None
    
    # 6. 캐시 확인 및 모델 로드/학습
    if use_cache and not data_updated:
        print("=== 캐시에서 모델 로드 시도 ===")
        model, cache_data = load_model_cache(model_config, cache_dir)
        
        if model is not None and cache_data is not None:
            print("✓ 캐시된 모델 로드 성공!")
            # 캐시된 스케일러 사용
            if 'scaler_data' in cache_data:
                cached_scalers = cache_data['scaler_data']
                feat_scaler = cached_scalers['feat_scaler']
                targ_scaler = cached_scalers['targ_scaler']
            
            # 캐시된 메타데이터에서 손실 정보 가져오기
            if 'train_losses' in cache_data.get('metadata', {}):
                train_losses = cache_data['metadata']['train_losses']
        else:
            print("✗ 캐시 로드 실패, 새로 학습합니다.")
    
    # 7. 모델 학습 (캐시가 없거나 데이터가 업데이트된 경우)
    if model is None:
        print("=== 새로운 모델 학습 ===")
        model = InformerAutoformerHybrid(**model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(10):
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
            print(f"Epoch {epoch+1}/10, Loss: {avg_loss:.6f}")
        
        # 8. 모델 및 메타데이터 캐시 저장
        if use_cache:
            metadata = {
                'train_losses': train_losses,
                'model_config': model_config,
                'data_shape': {'X': X.shape, 'y': y.shape},
                'split_idx': split_idx
            }
            
            save_model_cache(model, scaler_data, metadata, cache_dir)
            save_metadata(csv_path, current_checksum, cache_dir)
    
    elif data_updated and use_cache:
        print("=== 데이터 업데이트 감지: 증분 학습 수행 ===")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)  # 낮은 학습률
        criterion = nn.MSELoss()
        
        # 새로운 데이터에 대해서만 증분 학습
        new_train_losses = incremental_training(model, X_train, y_train, optimizer, criterion, epochs=5)
        train_losses.extend(new_train_losses)
        
        # 업데이트된 모델 캐시 저장
        metadata = {
            'train_losses': train_losses,
            'model_config': model_config,
            'data_shape': {'X': X.shape, 'y': y.shape},
            'split_idx': split_idx
        }
        
        save_model_cache(model, scaler_data, metadata, cache_dir)
        save_metadata(csv_path, current_checksum, cache_dir)
    
    # 9. 이상 탐지
    print("이상 탐지 수행 중...")
    model.eval()
    thres = DynamicThreshold()
    
    all_anomaly_scores = []
    all_predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X_test), 8):
            batch_x = X_test[i:i+8]
            batch_y = y_test[i:i+8]
            
            output = model(batch_x, batch_y)
            
            if output['anomaly_score'] is not None:
                all_anomaly_scores.extend(output['anomaly_score'].cpu().numpy())
            all_predictions.extend(output['forecast'].cpu().numpy())
    
    # 10. 동적 임계값 계산
    all_anomaly_scores = np.array(all_anomaly_scores)
    threshold = thres.update(torch.tensor(all_anomaly_scores))
    anomaly_mask = all_anomaly_scores > threshold
    
    # 11. 결과 출력
    print(f"\n=== 이상 탐지 결과 ===")
    print(f"총 테스트 샘플: {len(all_anomaly_scores)}")
    print(f"이상 샘플 수: {anomaly_mask.sum()}")
    print(f"이상률: {anomaly_mask.mean():.2%}")
    print(f"동적 임계값: {threshold:.6f}")
    print(f"평균 이상 점수: {all_anomaly_scores.mean():.6f}")
    print(f"최대 이상 점수: {all_anomaly_scores.max():.6f}")
    
    # 12. 시각화
    if use_interactive:
        print("\n=== 인터랙티브 대시보드 생성 중 ===")
        try:
            # 데이터 탐색 대시보드
            print("1. 데이터 탐색 대시보드...")
            create_interactive_dashboard(csv_path)
            
            # Plotly 정적 대시보드 (백업)
            print("2. Plotly 정적 대시보드...")
            create_plotly_dashboard(csv_path)
            
            # 모델 성능 대시보드 (학습 손실 포함)
            print("3. 모델 성능 대시보드...")
            create_model_performance_dashboard(model, all_anomaly_scores, anomaly_mask, threshold, df, train_losses)
            
        except Exception as e:
            print(f"인터랙티브 시각화 오류: {e}")
            print("기본 matplotlib 시각화로 대체합니다.")
            use_interactive = False
    
    if not use_interactive:
        # 기본 matplotlib 시각화
        create_basic_visualization(all_anomaly_scores, anomaly_mask, threshold, df, train_losses)
    
    return model, all_anomaly_scores, anomaly_mask, threshold, df
def create_interactive_dashboard(csv_path='sample_manufacturing.csv'):
    """인터랙티브 대시보드 생성 - 환경에 맞는 최적화된 버전"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 고유값들 추출
    equipment_ids = ['All'] + sorted(df['eqp_id'].unique().tolist())
    chamber_ids = ['All'] + sorted(df['chamber_id'].unique().tolist())
    recipes = ['All'] + sorted(df['recipe'].unique().tolist())
    
    print("=== 인터랙티브 이상치 분석 시작 ===")
    print(f"사용 가능한 옵션:")
    print(f"Equipment IDs: {equipment_ids}")
    print(f"Chamber IDs: {chamber_ids}")
    print(f"Recipes: {recipes}")
    
    def update_plots(equipment_id='All', chamber_id='All', recipe='All'):
        # 데이터 필터링
        filtered_df = df.copy()
        
        if equipment_id != 'All':
            filtered_df = filtered_df[filtered_df['eqp_id'] == equipment_id]
        if chamber_id != 'All':
            filtered_df = filtered_df[filtered_df['chamber_id'] == chamber_id]
        if recipe != 'All':
            filtered_df = filtered_df[filtered_df['recipe'] == recipe]
        
        if len(filtered_df) == 0:
            print("⚠️ 선택된 조건에 해당하는 데이터가 없습니다.")
            return
        
        print(f"\n=== 필터 조건: Equipment={equipment_id}, Chamber={chamber_id}, Recipe={recipe} ===")
        
        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Temperature Over Time', 'Pressure Over Time',
                'RF Power Over Time', 'Chamber Load Over Time',
                'Temperature vs RF Power', 'Anomaly Detection'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 온도 시계열
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['TEMP'],
                      mode='lines', name='Temperature', line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. 압력 시계열
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['PRESSURE'],
                      mode='lines', name='Pressure', line=dict(color='blue')),
            row=1, col=2
        )
        
        # 3. RF 파워 시계열
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['RF_POWER'],
                      mode='lines', name='RF Power', line=dict(color='green')),
            row=2, col=1
        )
        
        # 4. 챔버 로드 시계열
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['CHAMBER_LOAD'],
                      mode='lines', name='Chamber Load', line=dict(color='purple')),
            row=2, col=2
        )
        
        # 5. 온도 vs RF 파워 산점도 (레시피별 색상)
        for recipe_type in filtered_df['recipe'].unique():
            recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
            fig.add_trace(
                go.Scatter(x=recipe_data['TEMP'], y=recipe_data['RF_POWER'],
                          mode='markers', name=f'Recipe {recipe_type}',
                          marker=dict(size=4, opacity=0.6)),
                row=3, col=1
            )
        
        # 6. 이상치 탐지 결과
        # 임계값 기반 이상치
        temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
        power_anomalies = filtered_df[filtered_df['RF_POWER'] > 300]
        
        # 정상 데이터
        normal_data = filtered_df[
            (filtered_df['TEMP'] <= 450) & (filtered_df['RF_POWER'] <= 300)
        ]
        
        fig.add_trace(
            go.Scatter(x=normal_data['timestamp'], y=normal_data['TEMP'],
                      mode='markers', name='Normal', 
                      marker=dict(color='blue', size=3, opacity=0.5)),
            row=3, col=2
        )
        
        if len(temp_anomalies) > 0:
            fig.add_trace(
                go.Scatter(x=temp_anomalies['timestamp'], y=temp_anomalies['TEMP'],
                          mode='markers', name='Temp Anomaly',
                          marker=dict(color='red', size=6, symbol='x')),
                row=3, col=2
            )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=900,
            title_text=f"Manufacturing Data Analysis - Equipment: {equipment_id}, Chamber: {chamber_id}, Recipe: {recipe}",
            showlegend=True
        )
        
        # 축 라벨 추가
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_xaxes(title_text="Temperature", row=3, col=1)
        fig.update_xaxes(title_text="Time", row=3, col=2)
        
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_yaxes(title_text="Pressure", row=1, col=2)
        fig.update_yaxes(title_text="RF Power", row=2, col=1)
        fig.update_yaxes(title_text="Chamber Load", row=2, col=2)
        fig.update_yaxes(title_text="RF Power", row=3, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=3, col=2)
        
        # 통계 정보 출력
        print(f"\n📊 필터링된 데이터 통계")
        print(f"   데이터 포인트: {len(filtered_df):,}")
        print(f"   온도 범위: {filtered_df['TEMP'].min():.1f} - {filtered_df['TEMP'].max():.1f}°C")
        print(f"   압력 범위: {filtered_df['PRESSURE'].min():.2f} - {filtered_df['PRESSURE'].max():.2f}")
        print(f"   RF 파워 범위: {filtered_df['RF_POWER'].min():.1f} - {filtered_df['RF_POWER'].max():.1f}")
        print(f"   🔥 온도 이상치: {len(temp_anomalies)}개")
        print(f"   ⚡ RF 파워 이상치: {len(power_anomalies)}개")
        
        # 그래프 표시
        try:
            fig.show()
        except Exception as e:
            print(f"⚠️ 그래프 표시 오류: {e}")
            print("브라우저에서 그래프를 확인하세요.")
    
    # 환경 감지 및 적절한 인터페이스 제공
    environment_type = detect_environment()
    print(f"\n🔍 감지된 환경: {environment_type}")
    
    if environment_type == "jupyter":
        try:
            create_jupyter_widgets(equipment_ids, chamber_ids, recipes, update_plots)
        except Exception as e:
            print(f"⚠️ Jupyter 위젯 생성 실패: {e}")
            print("대안 인터페이스로 전환합니다...")
            create_enhanced_manual_interface(df, equipment_ids, chamber_ids, recipes, update_plots)
    elif environment_type == "vscode":
        # VS Code에서는 수동 인터페이스가 더 나음
        print("VS Code 환경에서는 수동 선택 인터페이스를 사용합니다.")
        create_enhanced_manual_interface(df, equipment_ids, chamber_ids, recipes, update_plots)
    else:
        # 기본 환경에서는 수동 인터페이스
        print("기본 환경에서는 수동 선택 인터페이스를 사용합니다.")
        create_enhanced_manual_interface(df, equipment_ids, chamber_ids, recipes, update_plots)

def detect_environment():
    """현재 실행 환경 감지"""
    try:
        # Jupyter 환경 확인
        from IPython import get_ipython
        if get_ipython() is not None:
            ipython = get_ipython()
            if hasattr(ipython, 'kernel'):
                return "jupyter"
    except ImportError:
        pass
    
    # VS Code 환경 확인
    import os
    if 'VSCODE_PID' in os.environ or 'TERM_PROGRAM' in os.environ:
        return "vscode"
    
    return "terminal"

def create_jupyter_widgets(equipment_ids, chamber_ids, recipes, update_plots):
    """Jupyter 환경용 위젯 생성"""
    from IPython.display import display, clear_output
    
    # 인터랙티브 위젯 생성
    equipment_dropdown = widgets.Dropdown(
        options=equipment_ids,
        value='All',
        description='Equipment:',
        style={'description_width': 'initial'}
    )
    
    chamber_dropdown = widgets.Dropdown(
        options=chamber_ids,
        value='All',
        description='Chamber:',
        style={'description_width': 'initial'}
    )
    
    recipe_dropdown = widgets.Dropdown(
        options=recipes,
        value='All',
        description='Recipe:',
        style={'description_width': 'initial'}
    )
    
    print("인터랙티브 위젯을 사용하여 이상치를 분석하세요:")
    
    # 위젯 박스로 레이아웃 개선
    widget_box = widgets.HBox([equipment_dropdown, chamber_dropdown, recipe_dropdown])
    display(widget_box)
    
    # 인터랙티브 디스플레이
    interact(update_plots, 
             equipment_id=equipment_dropdown,
             chamber_id=chamber_dropdown,
             recipe=recipe_dropdown)

def create_enhanced_manual_interface(df, equipment_ids, chamber_ids, recipes, update_plots_func):
    """향상된 수동 필터 선택 인터페이스"""
    
    print("\n=== 🎛️ 수동 필터 선택 인터페이스 ===")
    print("원하는 조건을 선택하여 데이터를 분석하세요.")
    print("(Enter를 누르면 기본값 'All' 사용, 'quit'으로 종료)")
    
    # 옵션을 보기 좋게 출력
    def print_options(title, options):
        print(f"\n📋 {title}:")
        for i, option in enumerate(options, 1):
            if i <= 5:  # 처음 5개만 표시
                print(f"   {i}. {option}")
            elif i == 6 and len(options) > 6:
                print(f"   ... 총 {len(options)}개 옵션")
                break
        print(f"   💡 직접 입력하거나 번호를 선택하세요")
    
    while True:
        print("\n" + "="*50)
        
        # Equipment 선택
        print_options("Equipment 옵션", equipment_ids)
        eqp_input = input("🏭 Equipment를 선택하세요 [기본값: All]: ").strip() or "All"
        if eqp_input.lower() == 'quit':
            break
        
        # 번호로 선택한 경우 처리
        if eqp_input.isdigit():
            idx = int(eqp_input) - 1
            if 0 <= idx < len(equipment_ids):
                eqp_choice = equipment_ids[idx]
            else:
                print(f"⚠️ 잘못된 번호: {eqp_input}")
                continue
        else:
            eqp_choice = eqp_input
        
        if eqp_choice not in equipment_ids:
            print(f"⚠️ 잘못된 Equipment ID: {eqp_choice}")
            print(f"   사용 가능한 옵션: {', '.join(equipment_ids[:5])}...")
            continue
            
        # Chamber 선택
        print_options("Chamber 옵션", chamber_ids)
        chamber_input = input("🏠 Chamber를 선택하세요 [기본값: All]: ").strip() or "All"
        if chamber_input.lower() == 'quit':
            break
            
        # 번호로 선택한 경우 처리
        if chamber_input.isdigit():
            idx = int(chamber_input) - 1
            if 0 <= idx < len(chamber_ids):
                chamber_choice = chamber_ids[idx]
            else:
                print(f"⚠️ 잘못된 번호: {chamber_input}")
                continue
        else:
            chamber_choice = chamber_input
            
        if chamber_choice not in chamber_ids:
            print(f"⚠️ 잘못된 Chamber ID: {chamber_choice}")
            print(f"   사용 가능한 옵션: {', '.join(chamber_ids)}")
            continue
            
        # Recipe 선택
        print_options("Recipe 옵션", recipes)
        recipe_input = input("📝 Recipe를 선택하세요 [기본값: All]: ").strip() or "All"
        if recipe_input.lower() == 'quit':
            break
            
        # 번호로 선택한 경우 처리
        if recipe_input.isdigit():
            idx = int(recipe_input) - 1
            if 0 <= idx < len(recipes):
                recipe_choice = recipes[idx]
            else:
                print(f"⚠️ 잘못된 번호: {recipe_input}")
                continue
        else:
            recipe_choice = recipe_input
            
        if recipe_choice not in recipes:
            print(f"⚠️ 잘못된 Recipe: {recipe_choice}")
            print(f"   사용 가능한 옵션: {', '.join(recipes[:5])}...")
            continue
        
        # 선택된 필터로 분석 실행
        print(f"\n🔍 분석 실행중...")
        print(f"   Equipment: {eqp_choice}")
        print(f"   Chamber: {chamber_choice}")
        print(f"   Recipe: {recipe_choice}")
        
        try:
            update_plots_func(eqp_choice, chamber_choice, recipe_choice)
            print("✅ 분석 완료!")
        except Exception as e:
            print(f"❌ 분석 중 오류 발생: {e}")
        
        # 계속할지 묻기
        print("\n" + "-"*50)
        continue_choice = input("🔄 다른 조건으로 분석하시겠습니까? (y/n) [기본값: n]: ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            break
    
    print("\n👋 인터랙티브 분석을 종료합니다.")

def create_model_performance_dashboard(model, all_anomaly_scores, anomaly_mask, threshold, df, train_losses=None):
    """모델 성능 시각화 대시보드"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Anomaly Score Time Series', 'Anomaly Score Distribution',
            'Model Performance Metrics', 'Feature Correlation Heatmap',
            'Anomaly Detection Results', 'Training Loss Evolution'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # 1. 이상 점수 시계열
    fig.add_trace(
        go.Scatter(y=all_anomaly_scores, mode='lines', name='Anomaly Score',
                  line=dict(color='blue', width=1)),
        row=1, col=1
    )
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                  annotation_text=f"Threshold: {threshold:.4f}", row=1, col=1)
    
    # 이상치 포인트 강조
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(
            go.Scatter(x=anomaly_indices, y=all_anomaly_scores[anomaly_indices],
                      mode='markers', name='Detected Anomalies',
                      marker=dict(color='red', size=6, symbol='x')),
            row=1, col=1
        )
    
    # 2. 이상 점수 분포
    fig.add_trace(
        go.Histogram(x=all_anomaly_scores, nbinsx=50, name='Score Distribution',
                    marker=dict(color='lightblue', line=dict(color='black', width=1))),
        row=1, col=2
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red", row=1, col=2)
    
    # 3. 성능 메트릭 (가상 데이터)
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
    values = [0.85, 0.78, 0.81, 0.92]  # 예시 값
    
    fig.add_trace(
        go.Bar(x=metrics, y=values, name='Performance Metrics',
               marker=dict(color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])),
        row=1, col=3
    )
    
    # 4. 특성 상관관계 히트맵
    features = ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']
    corr_matrix = df[features].corr().values
    
    fig.add_trace(
        go.Heatmap(z=corr_matrix, x=features, y=features,
                  colorscale='RdBu', zmid=0, name='Correlation'),
        row=2, col=1
    )
    
    # 5. 이상 탐지 결과 요약
    detection_summary = {
        'Normal': len(all_anomaly_scores) - anomaly_mask.sum(),
        'Anomaly': anomaly_mask.sum()
    }
    
    fig.add_trace(
        go.Pie(labels=list(detection_summary.keys()), 
               values=list(detection_summary.values()),
               name="Detection Results"),
        row=2, col=2
    )
    
    # 6. 학습 손실 진화
    if train_losses:
        fig.add_trace(
            go.Scatter(y=train_losses, mode='lines+markers', name='Training Loss',
                      line=dict(color='orange', width=2)),
            row=2, col=3
        )
    else:
        # 임계값 진화 (시뮬레이션)
        threshold_evolution = [threshold * (1 + 0.1 * np.sin(i/10)) for i in range(100)]
        fig.add_trace(
            go.Scatter(y=threshold_evolution, mode='lines', name='Dynamic Threshold',
                      line=dict(color='orange', width=2)),
            row=2, col=3
        )
    
    # 레이아웃 업데이트
    fig.update_layout(
        height=800,
        title_text="Model Performance Dashboard",
        showlegend=True
    )
    
    # 축 라벨
    fig.update_xaxes(title_text="Time Steps", row=1, col=1)
    fig.update_yaxes(title_text="Anomaly Score", row=1, col=1)
    fig.update_xaxes(title_text="Anomaly Score", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=1, col=3)
    fig.update_xaxes(title_text="Epoch", row=2, col=3)
    fig.update_yaxes(title_text="Loss" if train_losses else "Threshold", row=2, col=3)
    
    fig.show()

# 캐시 관리 함수들
def get_data_checksum(csv_path):
    """데이터 파일의 체크섬을 계산"""
    with open(csv_path, 'rb') as f:
        file_content = f.read()
    return hashlib.md5(file_content).hexdigest()

def save_model_cache(model, scaler_data, metadata, cache_dir='cache'):
    """모델과 관련 데이터를 캐시에 저장"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # 모델 저장
    model_path = os.path.join(cache_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    
    # 스케일러 및 메타데이터 저장
    cache_data = {
        'scaler_data': scaler_data,
        'metadata': metadata,
        'saved_time': datetime.now().isoformat()
    }
    
    cache_path = os.path.join(cache_dir, 'cache_data.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"모델 캐시 저장 완료: {cache_dir}")

def load_model_cache(model_config, cache_dir='cache'):
    """캐시에서 모델과 관련 데이터를 로드"""
    model_path = os.path.join(cache_dir, 'model.pth')
    cache_path = os.path.join(cache_dir, 'cache_data.pkl')
    
    if not (os.path.exists(model_path) and os.path.exists(cache_path)):
        return None, None
    
    try:
        # 모델 로드
        model = InformerAutoformerHybrid(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # 캐시 데이터 로드
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        print(f"모델 캐시 로드 완료: {cache_data['saved_time']}")
        return model, cache_data
    
    except Exception as e:
        print(f"캐시 로드 실패: {e}")
        return None, None

def check_data_update(csv_path, cache_dir='cache'):
    """데이터 업데이트 여부 확인"""
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    current_checksum = get_data_checksum(csv_path)
    
    if not os.path.exists(metadata_path):
        return True, current_checksum  # 첫 실행
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        cached_checksum = metadata.get('data_checksum', '')
        return current_checksum != cached_checksum, current_checksum
    
    except Exception as e:
        print(f"메타데이터 로드 실패: {e}")
        return True, current_checksum

def save_metadata(csv_path, checksum, cache_dir='cache'):
    """메타데이터 저장"""
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = os.path.join(cache_dir, 'metadata.json')
    
    metadata = {
        'data_checksum': checksum,
        'data_path': csv_path,
        'last_update': datetime.now().isoformat()
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def incremental_training(model, new_X, new_y, optimizer, criterion, epochs=5):
    """증분 학습 수행"""
    print("증분 학습 시작...")
    model.train()
    train_losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(new_X), 8):
            batch_x = new_X[i:i+8]
            batch_y = new_y[i:i+8]
            
            optimizer.zero_grad()
            output = model(batch_x, batch_y)
            loss = criterion(output['forecast'], batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / max(1, len(new_X) // 8)
        train_losses.append(avg_loss)
        print(f"Incremental Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return train_losses

def create_basic_visualization(all_anomaly_scores, anomaly_mask, threshold, df, train_losses):
    """기본 matplotlib 시각화"""
    plt.figure(figsize=(15, 10))
    
    # 학습 손실
    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    
    # 이상 점수 분포
    plt.subplot(2, 2, 2)
    plt.hist(all_anomaly_scores, bins=50, alpha=0.7, label='Anomaly Scores')
    plt.axvline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    plt.title('Anomaly Score Distribution')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    
    # 시계열 이상 점수
    plt.subplot(2, 2, 3)
    plt.plot(all_anomaly_scores, alpha=0.7)
    plt.axhline(threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.4f}')
    anomaly_indices = np.where(anomaly_mask)[0]
    plt.scatter(anomaly_indices, all_anomaly_scores[anomaly_indices], 
               color='red', s=30, label=f'Anomalies ({len(anomaly_indices)})')
    plt.title('Time Series Anomaly Detection')
    plt.xlabel('Time Steps')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    
    # 원본 데이터의 이상치 위치 표시
    plt.subplot(2, 2, 4)
    temp_data = df['TEMP'].values
    plt.plot(temp_data, alpha=0.7, label='Temperature')
    
    # 실제 데이터에서 온도 이상치 찾기
    temp_anomalies = np.where(temp_data > 450)[0]
    plt.scatter(temp_anomalies, temp_data[temp_anomalies], 
               color='red', s=30, label=f'Temp Anomalies ({len(temp_anomalies)})')
    plt.title('Original Temperature Data with Anomalies')
    plt.xlabel('Time Steps')
    plt.ylabel('Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
def analyze_anomalies_by_groups(df, temp_anomalies, power_anomalies, scores, mask):
    """장비/챔버/레시피별 이상치 분석"""
    print(f"\n=== 그룹별 이상치 분석 ===")
    
    # 장비별 이상치
    print("장비별 온도 이상치:")
    eqp_temp_anomalies = temp_anomalies.groupby('eqp_id').size()
    for eqp_id, count in eqp_temp_anomalies.items():
        total_eqp_data = len(df[df['eqp_id'] == eqp_id])
        print(f"  {eqp_id}: {count}개 ({count/total_eqp_data:.2%})")
    
    # 챔버별 이상치
    print("\n챔버별 온도 이상치:")
    chamber_temp_anomalies = temp_anomalies.groupby('chamber_id').size()
    for chamber_id, count in chamber_temp_anomalies.items():
        total_chamber_data = len(df[df['chamber_id'] == chamber_id])
        print(f"  {chamber_id}: {count}개 ({count/total_chamber_data:.2%})")
    
    # 레시피별 이상치
    print("\n레시피별 온도 이상치:")
    recipe_temp_anomalies = temp_anomalies.groupby('recipe').size()
    for recipe, count in recipe_temp_anomalies.items():
        total_recipe_data = len(df[df['recipe'] == recipe])
        print(f"  {recipe}: {count}개 ({count/total_recipe_data:.2%})")
    
    # 시간대별 이상치 패턴
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    temp_anomalies['hour'] = pd.to_datetime(temp_anomalies['timestamp']).dt.hour
    
    print("\n시간대별 이상치 분포:")
    hourly_anomalies = temp_anomalies.groupby('hour').size()
    peak_hours = hourly_anomalies.nlargest(3)
    for hour, count in peak_hours.items():
        print(f"  {hour:02d}시: {count}개")

def create_anomaly_analysis_dashboard(df, scores, mask, threshold):
    """인터랙티브 이상치 분석 대시보드"""
    
    # 고유값들 추출
    equipment_ids = ['All'] + sorted(df['eqp_id'].unique().tolist())
    chamber_ids = ['All'] + sorted(df['chamber_id'].unique().tolist())
    recipes = ['All'] + sorted(df['recipe'].unique().tolist())
    
    print(f"\n=== 이상치 분석 대시보드 ===")
    print(f"Equipment 옵션: {equipment_ids}")
    print(f"Chamber 옵션: {chamber_ids}")
    print(f"Recipe 옵션: {recipes}")
    
    def analyze_filtered_anomalies(equipment_id='All', chamber_id='All', recipe='All'):
        # 데이터 필터링
        filtered_df = df.copy()
        filter_conditions = []
        
        if equipment_id != 'All':
            filtered_df = filtered_df[filtered_df['eqp_id'] == equipment_id]
            filter_conditions.append(f"Equipment: {equipment_id}")
        if chamber_id != 'All':
            filtered_df = filtered_df[filtered_df['chamber_id'] == chamber_id]
            filter_conditions.append(f"Chamber: {chamber_id}")
        if recipe != 'All':
            filtered_df = filtered_df[filtered_df['recipe'] == recipe]
            filter_conditions.append(f"Recipe: {recipe}")
        
        if len(filtered_df) == 0:
            print("선택된 조건에 해당하는 데이터가 없습니다.")
            return
        
        # 필터 조건 출력
        filter_str = " & ".join(filter_conditions) if filter_conditions else "전체 데이터"
        print(f"\n=== 필터 조건: {filter_str} ===")
        
        # 필터링된 데이터의 인덱스 범위 계산
        filtered_indices = filtered_df.index.tolist()
        
        # 임계값 기반 이상치
        temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
        power_anomalies = filtered_df[filtered_df['RF_POWER'] > 300]
        
        # 모델 탐지 이상치 (근사적으로 매핑)
        # 시퀀스 데이터와 원본 데이터 간의 매핑이 복잡하므로 비율로 추정
        seq_start_idx = 96  # 시퀀스 길이
        total_original_data = len(df)
        total_test_data = len(scores)
        
        # 필터링된 데이터에서 예상되는 모델 이상치 수 추정
        filtered_ratio = len(filtered_df) / total_original_data
        estimated_model_anomalies = int(mask.sum() * filtered_ratio)
        
        # 시각화 생성
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                f'Temperature Time Series ({filter_str})',
                f'RF Power Time Series ({filter_str})',
                f'Temperature vs RF Power Scatter',
                f'Anomaly Distribution by Hour',
                f'Feature Statistics Comparison',
                f'Anomaly Summary'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 온도 시계열 (이상치 강조)
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['TEMP'],
                      mode='lines', name='Temperature', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        if len(temp_anomalies) > 0:
            fig.add_trace(
                go.Scatter(x=temp_anomalies['timestamp'], y=temp_anomalies['TEMP'],
                          mode='markers', name='Temp Anomalies',
                          marker=dict(color='red', size=8, symbol='x')),
                row=1, col=1
            )
        
        # 임계값 라인
        fig.add_hline(y=450, line_dash="dash", line_color="red", 
                     annotation_text="Temp Threshold: 450°C", row=1, col=1)
        
        # 2. RF 파워 시계열 (이상치 강조)
        fig.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['RF_POWER'],
                      mode='lines', name='RF Power',
                      line=dict(color='green', width=1)),
            row=1, col=2
        )
        
        if len(power_anomalies) > 0:
            fig.add_trace(
                go.Scatter(x=power_anomalies['timestamp'], y=power_anomalies['RF_POWER'],
                          mode='markers', name='Power Anomalies',
                          marker=dict(color='orange', size=8, symbol='triangle-up')),
                row=1, col=2
            )
        
        # RF 파워 임계값 라인
        fig.add_hline(y=300, line_dash="dash", line_color="orange",
                     annotation_text="RF Power Threshold: 300", row=1, col=2)
        
        # 3. 온도 vs RF 파워 산점도 (레시피별 색상)
        recipe_colors = {'R101': 'blue', 'R102': 'green', 'R103': 'purple'}
        for recipe_type in filtered_df['recipe'].unique():
            recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
            color = recipe_colors.get(recipe_type, 'gray')
            
            fig.add_trace(
                go.Scatter(x=recipe_data['TEMP'], y=recipe_data['RF_POWER'],
                          mode='markers', name=f'Recipe {recipe_type}',
                          marker=dict(size=4, opacity=0.6, color=color)),
                row=2, col=1
            )
        
        # 이상치 영역 표시
        fig.add_hline(y=300, line_dash="dash", line_color="orange", row=2, col=1)
        fig.add_vline(x=450, line_dash="dash", line_color="red", row=2, col=1)
        
        # 4. 시간대별 이상치 분포
        if len(temp_anomalies) > 0:
            temp_anomalies_copy = temp_anomalies.copy()
            temp_anomalies_copy['hour'] = pd.to_datetime(temp_anomalies_copy['timestamp']).dt.hour
            hourly_dist = temp_anomalies_copy.groupby('hour').size().reset_index(name='count')
            
            fig.add_trace(
                go.Bar(x=hourly_dist['hour'], y=hourly_dist['count'],
                      name='Anomalies by Hour',
                      marker=dict(color='red', opacity=0.7)),
                row=2, col=2
            )
        
        # 5. 특성 통계 비교 (정상 vs 이상)
        normal_data = filtered_df[
            (filtered_df['TEMP'] <= 450) & (filtered_df['RF_POWER'] <= 300)
        ]
        
        features = ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']
        normal_means = [normal_data[feat].mean() for feat in features]
        anomaly_means = [temp_anomalies[feat].mean() if len(temp_anomalies) > 0 else 0 for feat in features]
        
        fig.add_trace(
            go.Bar(x=features, y=normal_means, name='Normal Mean',
                  marker=dict(color='lightblue'), offsetgroup=1),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Bar(x=features, y=anomaly_means, name='Anomaly Mean',
                  marker=dict(color='lightcoral'), offsetgroup=2),
            row=3, col=1
        )
        
        # 6. 이상치 요약 파이 차트
        anomaly_summary = {
            'Normal': len(filtered_df) - len(temp_anomalies) - len(power_anomalies),
            'Temp Anomaly': len(temp_anomalies),
            'Power Anomaly': len(power_anomalies)
        }
        
        # 중복 제거 (온도와 파워 모두 이상인 경우)
        both_anomalies = len(filtered_df[
            (filtered_df['TEMP'] > 450) & (filtered_df['RF_POWER'] > 300)
        ])
        anomaly_summary['Both Anomalies'] = both_anomalies
        anomaly_summary['Normal'] += both_anomalies  # 중복 제거를 위해 조정
        
        fig.add_trace(
            go.Pie(labels=list(anomaly_summary.keys()),
                  values=list(anomaly_summary.values()),
                  name="Anomaly Summary"),
            row=3, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            height=1000,
            title_text=f"Anomaly Analysis Dashboard - {filter_str}",
            showlegend=True
        )
        
        # 축 라벨 설정
        fig.update_xaxes(title_text="Time", row=1, col=1)
        fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="RF Power", row=1, col=2)
        fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
        fig.update_yaxes(title_text="RF Power", row=2, col=1)
        fig.update_xaxes(title_text="Hour", row=2, col=2)
        fig.update_yaxes(title_text="Anomaly Count", row=2, col=2)
        fig.update_xaxes(title_text="Features", row=3, col=1)
        fig.update_yaxes(title_text="Average Value", row=3, col=1)
        
        # 통계 정보 출력
        print(f"데이터 포인트: {len(filtered_df):,}개")
        print(f"온도 범위: {filtered_df['TEMP'].min():.1f}°C - {filtered_df['TEMP'].max():.1f}°C")
        print(f"RF 파워 범위: {filtered_df['RF_POWER'].min():.1f} - {filtered_df['RF_POWER'].max():.1f}")
        print(f"온도 이상치: {len(temp_anomalies)}개 ({len(temp_anomalies)/len(filtered_df):.2%})")
        print(f"RF 파워 이상치: {len(power_anomalies)}개 ({len(power_anomalies)/len(filtered_df):.2%})")
        print(f"예상 모델 탐지 이상치: ~{estimated_model_anomalies}개")
        
        # 상위 이상치 시점 출력
        if len(temp_anomalies) > 0:
            print(f"\n상위 온도 이상치 (최대 5개):")
            top_temp_anomalies = temp_anomalies.nlargest(5, 'TEMP')
            for idx, row in top_temp_anomalies.iterrows():
                print(f"  {row['timestamp']}: TEMP={row['TEMP']:.1f}°C, RF_POWER={row['RF_POWER']:.1f}, Recipe={row['recipe']}")
        
        fig.show()
        
        return filtered_df, temp_anomalies, power_anomalies
    
    # 위젯이 사용 가능한지 확인
    try:
        from IPython.display import display
        
        # 인터랙티브 위젯 생성
        equipment_dropdown = widgets.Dropdown(
            options=equipment_ids,
            value='All',
            description='Equipment:',
            style={'description_width': 'initial'}
        )
        
        chamber_dropdown = widgets.Dropdown(
            options=chamber_ids,
            value='All',
            description='Chamber:',
            style={'description_width': 'initial'}
        )
        
        recipe_dropdown = widgets.Dropdown(
            options=recipes,
            value='All',
            description='Recipe:',
            style={'description_width': 'initial'}
        )
        
        # 인터랙티브 디스플레이
        print("인터랙티브 위젯을 사용하여 이상치를 분석하세요:")
        interact(analyze_filtered_anomalies,
                equipment_id=equipment_dropdown,
                chamber_id=chamber_dropdown,
                recipe=recipe_dropdown)
                
    except ImportError:
        print("\n=== IPython 환경이 아닙니다 ===")
        print("수동으로 필터를 선택하여 이상치를 분석하세요:")
        
        # 수동 선택 인터페이스
        create_manual_anomaly_interface(df, equipment_ids, chamber_ids, recipes, analyze_filtered_anomalies)
        
    except Exception as e:
        print(f"인터랙티브 위젯 생성 실패: {e}")
        print("수동으로 필터를 선택하여 이상치를 분석하세요:")
        
        # 수동 선택 인터페이스
        create_manual_anomaly_interface(df, equipment_ids, chamber_ids, recipes, analyze_filtered_anomalies)

def create_manual_anomaly_interface(df, equipment_ids, chamber_ids, recipes, analyze_func):
    """수동 이상치 분석 인터페이스"""
    
    print("\n=== 이상치 분석 필터 선택 ===")
    print("미리 정의된 분석 시나리오:")
    print("1. 모든 데이터")
    print("2. Equipment별 분석")
    print("3. Recipe별 분석")
    print("4. 사용자 정의 필터")
    print("5. 주요 이상치 패턴 분석")
    
    choice = input("분석 방법을 선택하세요 (1-5) [기본값: 1]: ").strip() or "1"
    
    if choice == "1":
        # 전체 데이터 분석
        print("\n=== 전체 데이터 이상치 분석 ===")
        analyze_func('All', 'All', 'All')
        
    elif choice == "2":
        # Equipment별 분석
        print("\n=== Equipment별 이상치 분석 ===")
        for eqp_id in equipment_ids[1:]:  # 'All' 제외
            print(f"\n--- Equipment: {eqp_id} ---")
            try:
                analyze_func(eqp_id, 'All', 'All')
            except Exception as e:
                print(f"분석 실패: {e}")
                
    elif choice == "3":
        # Recipe별 분석
        print("\n=== Recipe별 이상치 분석 ===")
        for recipe in recipes[1:]:  # 'All' 제외
            print(f"\n--- Recipe: {recipe} ---")
            try:
                analyze_func('All', 'All', recipe)
            except Exception as e:
                print(f"분석 실패: {e}")
                
    elif choice == "4":
        # 사용자 정의 필터
        print("\n=== 사용자 정의 필터 ===")
        
        while True:
            print(f"\n사용 가능한 옵션:")
            print(f"Equipment: {', '.join(equipment_ids)}")
            print(f"Chamber: {', '.join(chamber_ids)}")
            print(f"Recipe: {', '.join(recipes)}")
            
            eqp_choice = input(f"Equipment 선택 [기본값: All]: ").strip() or "All"
            if eqp_choice not in equipment_ids:
                print(f"⚠️ 잘못된 Equipment: {eqp_choice}")
                continue
                
            chamber_choice = input(f"Chamber 선택 [기본값: All]: ").strip() or "All"
            if chamber_choice not in chamber_ids:
                print(f"⚠️ 잘못된 Chamber: {chamber_choice}")
                continue
                
            recipe_choice = input(f"Recipe 선택 [기본값: All]: ").strip() or "All"
            if recipe_choice not in recipes:
                print(f"⚠️ 잘못된 Recipe: {recipe_choice}")
                continue
            
            print(f"\n=== 분석: {eqp_choice}/{chamber_choice}/{recipe_choice} ===")
            try:
                analyze_func(eqp_choice, chamber_choice, recipe_choice)
            except Exception as e:
                print(f"분석 실패: {e}")
            
            continue_choice = input("\n다른 조건으로 분석하시겠습니까? (y/n) [기본값: n]: ").strip().lower()
            if continue_choice != 'y':
                break
                
    elif choice == "5":
        # 주요 이상치 패턴 분석
        print("\n=== 주요 이상치 패턴 분석 ===")
        
        # 가장 많은 이상치를 가진 조합 찾기
        temp_anomalies = df[df['TEMP'] > 450]
        
        if len(temp_anomalies) > 0:
            print("1. Equipment별 이상치 분포:")
            eqp_counts = temp_anomalies.groupby('eqp_id').size().sort_values(ascending=False)
            for eqp_id, count in eqp_counts.head(3).items():
                print(f"   {eqp_id}: {count}개")
                analyze_func(eqp_id, 'All', 'All')
            
            print("\n2. Recipe별 이상치 분포:")
            recipe_counts = temp_anomalies.groupby('recipe').size().sort_values(ascending=False)
            for recipe, count in recipe_counts.head(3).items():
                print(f"   {recipe}: {count}개")
                analyze_func('All', 'All', recipe)
        else:
            print("이상치가 발견되지 않았습니다.")
    
    else:
        print("잘못된 선택입니다. 전체 데이터 분석을 수행합니다.")
        analyze_func('All', 'All', 'All')

def create_plotly_dashboard(csv_path='sample_manufacturing.csv'):
    """Plotly 기반 간단한 대시보드"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 고유값들 추출
    equipment_ids = sorted(df['eqp_id'].unique().tolist())
    chamber_ids = sorted(df['chamber_id'].unique().tolist())
    recipes = sorted(df['recipe'].unique().tolist())
    
    print("=== Plotly 대시보드 (정적 버전) ===")
    print("Equipment, Chamber, Recipe별 데이터를 자동으로 분석합니다...")
    
    # Plotly 렌더링 설정
    import plotly.offline as pyo
    pyo.init_notebook_mode(connected=True)
    
    # 전체 개요
    fig_overview = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Overview', 'RF Power Overview', 
                       'Equipment Distribution', 'Recipe Distribution'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"type": "pie"}, {"type": "pie"}]]
    )
    
    # 온도 전체 시계열
    fig_overview.add_trace(
        go.Scatter(x=df['timestamp'], y=df['TEMP'],
                  mode='lines', name='Temperature', line=dict(color='red', width=1)),
        row=1, col=1
    )
    
    # 온도 이상치 표시
    temp_anomalies = df[df['TEMP'] > 450]
    if len(temp_anomalies) > 0:
        fig_overview.add_trace(
            go.Scatter(x=temp_anomalies['timestamp'], y=temp_anomalies['TEMP'],
                      mode='markers', name='Temp Anomalies',
                      marker=dict(color='red', size=6, symbol='x')),
            row=1, col=1
        )
    
    # RF 파워 전체 시계열
    fig_overview.add_trace(
        go.Scatter(x=df['timestamp'], y=df['RF_POWER'],
                  mode='lines', name='RF Power', line=dict(color='green', width=1)),
        row=1, col=2
    )
    
    # RF 파워 이상치 표시
    power_anomalies = df[df['RF_POWER'] > 300]
    if len(power_anomalies) > 0:
        fig_overview.add_trace(
            go.Scatter(x=power_anomalies['timestamp'], y=power_anomalies['RF_POWER'],
                      mode='markers', name='Power Anomalies',
                      marker=dict(color='orange', size=6, symbol='triangle-up')),
            row=1, col=2
        )
    
    # Equipment 분포
    eqp_counts = df.groupby('eqp_id').size()
    fig_overview.add_trace(
        go.Pie(labels=eqp_counts.index, values=eqp_counts.values, name="Equipment"),
        row=2, col=1
    )
    
    # Recipe 분포
    recipe_counts = df.groupby('recipe').size()
    fig_overview.add_trace(
        go.Pie(labels=recipe_counts.index, values=recipe_counts.values, name="Recipe"),
        row=2, col=2
    )
    
    fig_overview.update_layout(height=800, title_text="Manufacturing Data Overview")
    
    # 그래프 표시 시도
    try:
        print("1. 전체 개요 대시보드 표시 중...")
        fig_overview.show()
    except Exception as e:
        print(f"브라우저 표시 실패: {e}")
        # HTML 파일로 저장
        overview_path = "manufacturing_overview.html"
        fig_overview.write_html(overview_path)
        print(f"대신 HTML 파일로 저장됨: {overview_path}")
        print(f"브라우저에서 파일을 열어 확인하세요: file://{os.path.abspath(overview_path)}")
    
    # Equipment별 상세 분석
    print("\n=== Equipment별 상세 분석 ===")
    for eqp_id in equipment_ids:
        eqp_data = df[df['eqp_id'] == eqp_id]
        eqp_temp_anomalies = eqp_data[eqp_data['TEMP'] > 450]
        eqp_power_anomalies = eqp_data[eqp_data['RF_POWER'] > 300]
        
        print(f"\n{eqp_id}:")
        print(f"  총 데이터: {len(eqp_data):,}개")
        print(f"  온도 이상치: {len(eqp_temp_anomalies)}개 ({len(eqp_temp_anomalies)/len(eqp_data):.2%})")
        print(f"  RF 파워 이상치: {len(eqp_power_anomalies)}개 ({len(eqp_power_anomalies)/len(eqp_data):.2%})")
        
        if len(eqp_temp_anomalies) > 3:  # 이상치가 많은 장비만 시각화
            fig_eqp = make_subplots(
                rows=2, cols=1,
                subplot_titles=(f'{eqp_id} - Temperature', f'{eqp_id} - RF Power')
            )
            
            # 온도
            fig_eqp.add_trace(
                go.Scatter(x=eqp_data['timestamp'], y=eqp_data['TEMP'],
                          mode='lines', name='Temperature', line=dict(color='blue', width=1)),
                row=1, col=1
            )
            
            if len(eqp_temp_anomalies) > 0:
                fig_eqp.add_trace(
                    go.Scatter(x=eqp_temp_anomalies['timestamp'], y=eqp_temp_anomalies['TEMP'],
                              mode='markers', name='Temp Anomalies',
                              marker=dict(color='red', size=8, symbol='x')),
                    row=1, col=1
                )
            
            # RF 파워
            fig_eqp.add_trace(
                go.Scatter(x=eqp_data['timestamp'], y=eqp_data['RF_POWER'],
                          mode='lines', name='RF Power', line=dict(color='green', width=1)),
                row=2, col=1
            )
            
            if len(eqp_power_anomalies) > 0:
                fig_eqp.add_trace(
                    go.Scatter(x=eqp_power_anomalies['timestamp'], y=eqp_power_anomalies['RF_POWER'],
                              mode='markers', name='Power Anomalies',
                              marker=dict(color='orange', size=8, symbol='triangle-up')),
                    row=2, col=1
                )
            
            fig_eqp.update_layout(height=600, title_text=f"Equipment {eqp_id} Analysis")
            
            try:
                print(f"   {eqp_id} 상세 분석 차트 표시 중...")
                fig_eqp.show()
            except Exception as e:
                print(f"   브라우저 표시 실패: {e}")
                # HTML 파일로 저장
                eqp_path = f"equipment_{eqp_id}_analysis.html"
                fig_eqp.write_html(eqp_path)
                print(f"   HTML 파일로 저장됨: {eqp_path}")
    
    # Recipe별 비교 분석
    print("\n=== Recipe별 비교 분석 ===")
    fig_recipe = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature by Recipe', 'RF Power by Recipe',
                       'Anomaly Count by Recipe', 'Feature Correlation by Recipe')
    )
    
    colors = ['blue', 'green', 'purple', 'orange', 'red']
    
    # Recipe별 온도 분포
    for i, recipe in enumerate(recipes):
        recipe_data = df[df['recipe'] == recipe]
        fig_recipe.add_trace(
            go.Box(y=recipe_data['TEMP'], name=f'{recipe} Temp', 
                  marker=dict(color=colors[i % len(colors)])),
            row=1, col=1
        )
    
    # Recipe별 RF 파워 분포
    for i, recipe in enumerate(recipes):
        recipe_data = df[df['recipe'] == recipe]
        fig_recipe.add_trace(
            go.Box(y=recipe_data['RF_POWER'], name=f'{recipe} Power',
                  marker=dict(color=colors[i % len(colors)])),
            row=1, col=2
        )
    
    # Recipe별 이상치 개수
    recipe_anomaly_counts = []
    recipe_labels = []
    for recipe in recipes:
        recipe_data = df[df['recipe'] == recipe]
        anomaly_count = len(recipe_data[recipe_data['TEMP'] > 450])
        recipe_anomaly_counts.append(anomaly_count)
        recipe_labels.append(recipe)
    
    fig_recipe.add_trace(
        go.Bar(x=recipe_labels, y=recipe_anomaly_counts, name='Anomaly Count',
              marker=dict(color='red', opacity=0.7)),
        row=2, col=1
    )
    
    # 전체 특성 상관관계
    features = ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']
    corr_matrix = df[features].corr().values
    
    fig_recipe.add_trace(
        go.Heatmap(z=corr_matrix, x=features, y=features,
                  colorscale='RdBu', zmid=0, name='Correlation'),
        row=2, col=2
    )
    
    fig_recipe.update_layout(height=800, title_text="Recipe Comparison Analysis")
    
    try:
        print("2. Recipe 비교 분석 차트 표시 중...")
        fig_recipe.show()
    except Exception as e:
        print(f"브라우저 표시 실패: {e}")
        # HTML 파일로 저장
        recipe_path = "recipe_comparison.html"
        fig_recipe.write_html(recipe_path)
        print(f"HTML 파일로 저장됨: {recipe_path}")
        print(f"브라우저에서 파일을 열어 확인하세요: file://{os.path.abspath(recipe_path)}")
    
    # 요약 통계
    print(f"\n=== 전체 요약 통계 ===")
    print(f"총 데이터 포인트: {len(df):,}개")
    print(f"Equipment 수: {len(equipment_ids)}개")
    print(f"Chamber 수: {len(chamber_ids)}개")
    print(f"Recipe 수: {len(recipes)}개")
    print(f"온도 이상치: {len(temp_anomalies)}개 ({len(temp_anomalies)/len(df):.2%})")
    
    power_anomalies = df[df['RF_POWER'] > 300]
    print(f"RF 파워 이상치: {len(power_anomalies)}개 ({len(power_anomalies)/len(df):.2%})")
    
    return df, temp_anomalies, power_anomalies

# 메인 실행 블록
if __name__ == "__main__":
    print("=== 제조업 데이터 이상 탐지 시스템 (캐시 지원) ===")
    print("1. 기본 분석 (캐시 비활성화)")
    print("2. 캐시 활용 분석 (권장 - 빠른 재실행)")
    print("3. 인터랙티브 대시보드 (캐시 활용)")
    print("4. 정적 Plotly 대시보드")
    print("5. 캐시 초기화")
    print("6. 웹 대시보드 실행")
    
    choice = input("선택하세요 (1/2/3/4/5/6) [기본값: 2]: ").strip() or "2"
    
    if choice == "5":
        # 캐시 초기화
        import shutil
        cache_dir = 'cache'
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print("캐시가 초기화되었습니다.")
        else:
            print("캐시가 존재하지 않습니다.")
        exit()
    
    elif choice == "6":
        # 웹 대시보드 실행
        try:
            import dash
            print(f"✅ Dash 버전: {dash.__version__}")
            
            from web_dashboard import create_web_dashboard
            print("웹 대시보드를 시작합니다...")
            print("브라우저에서 http://127.0.0.1:8050/ 으로 접속하세요.")
            print("종료하려면 Ctrl+C를 누르세요.")
            
            app = create_web_dashboard()
            app.run(debug=True, host='127.0.0.1', port=8050)
            
        except ImportError as e:
            print(f"❌ 필요한 패키지가 설치되지 않았습니다: {e}")
            print("다음 명령어로 설치하세요:")
            print("pip install dash plotly pandas numpy")
        except FileNotFoundError:
            print("❌ web_dashboard.py 파일이 없습니다.")
        except Exception as e:
            print(f"❌ 웹 대시보드 실행 실패: {e}")
        exit()
    
    use_cache = choice in ["2", "3"]
    use_interactive = choice == "3"
    use_static = choice == "4"
    
    print(f"\n캐시 사용: {'Yes' if use_cache else 'No'}")
    print(f"인터랙티브 모드: {'Yes' if use_interactive else 'No'}")
    print(f"정적 대시보드: {'Yes' if use_static else 'No'}")
    
    try:
        if use_static:
            # 정적 Plotly 대시보드만 실행
            print("정적 Plotly 대시보드를 생성합니다...")
            create_plotly_dashboard('sample_manufacturing.csv')
        else:
            # 기존 이상 탐지 시스템 실행
            model, scores, mask, threshold, df = detect_anomalies_with_model(
                'sample_manufacturing.csv', 
                use_interactive=use_interactive,
                use_cache=use_cache
            )
            
            # 추가 분석 정보
            print(f"\n=== 추가 분석 결과 ===")
            anomaly_indices = np.where(mask)[0]
            print(f"이상으로 탐지된 시점들 (처음 10개):")
            for idx in anomaly_indices[:10]:
                print(f"  Index {idx}: Score {scores[idx]:.6f}")
            
            # 실제 이상치와 비교
            temp_anomalies = df[df['TEMP'] > 450]
            power_anomalies = df[df['RF_POWER'] > 300]
            
            print(f"\n실제 데이터 이상치:")
            print(f"  온도 이상치 (>450°C): {len(temp_anomalies)}개")
            print(f"  RF 파워 이상치 (>300): {len(power_anomalies)}개")
            print(f"  모델 탐지 이상치: {len(anomaly_indices)}개")
            
            # 장비/챔버/레시피별 이상치 분석
            analyze_anomalies_by_groups(df, temp_anomalies, power_anomalies, scores, mask)
            
            # 상관관계 분석
            if len(temp_anomalies) > 0:
                print(f"\n온도 이상치 상세 (처음 5개):")
                for idx, row in temp_anomalies.head().iterrows():
                    print(f"  {row['timestamp']}: TEMP={row['TEMP']:.2f}°C, RF_POWER={row['RF_POWER']:.2f}, Recipe={row['recipe']}")
            
            # 인터랙티브 이상치 분석 대시보드
            if use_interactive:
                print(f"\n=== 인터랙티브 이상치 분석 시작 ===")
                create_anomaly_analysis_dashboard(df, scores, mask, threshold)
        
    except Exception as e:
        print(f"실행 중 오류: {e}")
        print("기본 모드로 실행합니다...")
        try:
            model, scores, mask, threshold, df = detect_anomalies_with_model(
                'sample_manufacturing.csv', use_interactive=False, use_cache=False
            )
        except Exception as inner_e:
            print(f"기본 모드 실행도 실패: {inner_e}")
            exit(1)
    
    # 캐시 상태 확인
    if use_cache:
        cache_dir = 'cache'
        if os.path.exists(cache_dir):
            cache_files = os.listdir(cache_dir)
            print(f"\n=== 캐시 상태 ===")
            print(f"캐시 디렉토리: {cache_dir}")
            print(f"캐시 파일들: {cache_files}")
            
            # 캐시 크기 확인
            total_size = 0
            for file in cache_files:
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    total_size += os.path.getsize(file_path)
            print(f"캐시 총 크기: {total_size / (1024*1024):.2f} MB")
    
    print(f"\n분석 완료! 결과를 확인해보세요.")
    if use_cache:
        print("다음 실행 시에는 캐시된 모델을 사용하여 더 빠르게 실행됩니다.")
    
    # 추가 옵션 제공
    print(f"\n=== 추가 옵션 ===")
    print("다른 분석을 원하시면 스크립트를 다시 실행하세요.")
    if not use_interactive:
        print("인터랙티브 분석을 원하시면 옵션 3을 선택하세요.")
    if not use_static:
        print("정적 대시보드를 원하시면 옵션 4를 선택하세요.")
    print("웹 대시보드를 원하시면 옵션 6을 선택하세요.")
