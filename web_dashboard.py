import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ================== Heuristic 회귀분석 파라미터 (heuristic.py와 동일) ==================
MSE_THRESHOLD = 0.1
SLOPE_DIFF_THRESHOLD = 0.25
WINDOW_SIZE = 30

def heuristic_alpha(X, y):
    """heuristic.py와 동일한 alpha 계산"""
    y_std = np.std(y)
    x_range = X['x'].max() - X['x'].min()
    if x_range == 0:
        return 0.01
    scale = y_std / x_range
    return max(min(scale, 1.0), 0.01)

def evaluate_merge_condition(data, idx1, idx2):
    """heuristic.py와 동일한 병합 조건 평가"""
    df_combined = data.iloc[idx1[0]:idx2[1]]
    if df_combined.isnull().any().any() or len(df_combined) < 2:
        return False
    alpha = heuristic_alpha(df_combined[['x']], df_combined['y'])
    try:
        model = Lasso(alpha=alpha).fit(df_combined[['x']], df_combined['y'])
        preds = model.predict(df_combined[['x']])
        mse = mean_squared_error(df_combined['y'], preds)
        var = np.var(df_combined['y'])
        mse_norm = mse / var if var > 0 else mse

        df1 = data.iloc[idx1[0]:idx1[1]]
        df2 = data.iloc[idx2[0]:idx2[1]]
        m1 = Lasso(alpha=alpha).fit(df1[['x']], df1['y']).coef_[0]
        m2 = Lasso(alpha=alpha).fit(df2[['x']], df2['y']).coef_[0]

        # 추가조건1: slope 변화량 기준
        slope_diffs = np.diff(preds)
        slope_var = np.var(slope_diffs)

        # 추가조건2: 예측값 곡률
        if np.std(df_combined['y'] - preds) < 1.0:
            curvature = np.std(np.diff(preds, n=2))
            if curvature > 0.05:
                return False

        return (mse_norm < MSE_THRESHOLD) and (abs(m1 - m2) < SLOPE_DIFF_THRESHOLD) and (slope_var < 0.05)
    except:
        return False

def generate_initial_segments(data):
    """heuristic.py와 동일한 초기 세그먼트 생성"""
    return [(i, i + WINDOW_SIZE) for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE)]

def bottom_up_merge(data, segments):
    """heuristic.py와 동일한 bottom-up 병합"""
    merged = []
    i = 0
    while i < len(segments):
        if i < len(segments) - 1 and evaluate_merge_condition(data, segments[i], segments[i+1]):
            segments[i+1] = (segments[i][0], segments[i+1][1])
        else:
            merged.append(segments[i])
        i += 1
    return merged

def process_segment_label(seg, data):
    """heuristic.py와 동일한 세그먼트 라벨링"""
    df_seg = data.iloc[seg[0]:seg[1]]
    if df_seg.isnull().any().any() or len(df_seg) < 2:
        return {'segment': seg, 'alpha': None, 'slope': np.nan, 'slope_sign': 'N/A', 'linearity': 'N/A', 'pred': np.full(len(df_seg), np.nan), 'p_value': np.nan, 'r_squared': np.nan}
    
    alpha = heuristic_alpha(df_seg[['x']], df_seg['y'])
    model = Lasso(alpha=alpha).fit(df_seg[['x']], df_seg['y'])
    pred = model.predict(df_seg[['x']])
    slope = model.coef_[0]

    # heuristic.py와 동일한 slope 분류 (0.05 임계값)
    if abs(slope) <= 0.05:
        slope_sign = 'stable'
    elif slope > 0:
        slope_sign = 'increase'
    else:
        slope_sign = 'decrease'

    # heuristic.py와 동일한 선형성 검정 (p-value <= 0.05, R² >= 0.3)
    try:
        X_const = sm.add_constant(df_seg[['x']])
        ols = sm.OLS(df_seg['y'], X_const).fit()
        p_value = ols.pvalues[1]
        r_squared = ols.rsquared
        linearity = 'linear' if (p_value <= 0.05 and r_squared >= 0.3) else 'non-linear'
    except:
        p_value = np.nan
        r_squared = np.nan
        linearity = 'unknown'

    return {
        'segment': seg, 
        'alpha': alpha, 
        'slope': slope, 
        'slope_sign': slope_sign, 
        'linearity': linearity, 
        'pred': pred,
        'p_value': p_value,
        'r_squared': r_squared
    }

def analyze_timeseries_regression(df, feature_name):
    """시계열 데이터에 heuristic.py와 동일한 회귀분석 적용"""
    if len(df) < WINDOW_SIZE:
        return []
    
    # heuristic.py와 동일한 데이터 구조 생성
    data = pd.DataFrame({'x': df.index, 'y': df[feature_name]})
    
    # heuristic.py와 동일한 세그먼트 생성 및 병합
    segments = generate_initial_segments(data)
    if not segments:
        return []
    
    merged_segments = bottom_up_merge(data, segments)
    
    # 각 세그먼트 분석
    results = []
    for seg in merged_segments:
        result = process_segment_label(seg, data)
        if result['alpha'] is not None:
            # 원본 timestamp 및 값 정보 추가
            start_idx = seg[0]
            end_idx = min(seg[1], len(df))
            result['start_time'] = df.iloc[start_idx]['timestamp']
            result['end_time'] = df.iloc[end_idx-1]['timestamp'] 
            result['start_value'] = df.iloc[start_idx][feature_name]
            result['end_value'] = df.iloc[end_idx-1][feature_name]
            result['segment_length'] = end_idx - start_idx
            results.append(result)
    
    return results

def create_web_dashboard(csv_path='sample_manufacturing.csv'):
    """웹 기반 대시보드 생성"""
    
    # 데이터 로드
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Dash 앱 초기화
    app = dash.Dash(__name__)
    
    # 고유값들 추출
    equipment_ids = sorted(df['eqp_id'].unique())
    chamber_ids = sorted(df['chamber_id'].unique())
    recipes = sorted(df['recipe'].unique())
    
    # 레이아웃 정의
    app.layout = html.Div([
        html.H1("제조업 데이터 분석 대시보드", 
                style={'text-align': 'center', 'margin-bottom': '30px'}),
        
        # 컨트롤 패널
        html.Div([
            html.Div([
                html.Label("장비 ID:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='equipment-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': eqp, 'value': eqp} for eqp in equipment_ids],
                    value='All',
                    style={'margin-bottom': '10px'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '5%'}),
            
            html.Div([
                html.Label("챔버 ID:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='chamber-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': ch, 'value': ch} for ch in chamber_ids],
                    value='All',
                    style={'margin-bottom': '10px'}
                )
            ], style={'width': '30%', 'display': 'inline-block', 'margin-right': '5%'}),
            
            html.Div([
                html.Label("레시피:", style={'font-weight': 'bold'}),
                dcc.Dropdown(
                    id='recipe-dropdown',
                    options=[{'label': 'All', 'value': 'All'}] + 
                            [{'label': recipe, 'value': recipe} for recipe in recipes],
                    value='All',
                    style={'margin-bottom': '10px'}
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
        ], style={'margin-bottom': '30px', 'padding': '20px', 'background-color': '#f8f9fa', 'border-radius': '10px'}),
        
        # 통계 요약 패널
        html.Div(id='stats-panel', style={'margin-bottom': '20px'}),
        
        # 메인 차트들
        html.Div([
            dcc.Graph(id='time-series-chart'),
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='correlation-chart'),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='anomaly-chart'),
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'}),
        ], style={'margin-bottom': '20px'}),
        
        html.Div([
            html.Div([
                dcc.Graph(id='distribution-chart'),
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(id='recipe-comparison-chart'),
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%'}),
        ])
    ])
    
    # 콜백 함수들
    @app.callback(
        [Output('stats-panel', 'children'),
         Output('time-series-chart', 'figure'),
         Output('correlation-chart', 'figure'),
         Output('anomaly-chart', 'figure'),
         Output('distribution-chart', 'figure'),
         Output('recipe-comparison-chart', 'figure')],
        [Input('equipment-dropdown', 'value'),
         Input('chamber-dropdown', 'value'),
         Input('recipe-dropdown', 'value')]
    )
    def update_dashboard(equipment_id, chamber_id, recipe):
        try:
            # 데이터 필터링
            filtered_df = df.copy()
            
            if equipment_id != 'All':
                filtered_df = filtered_df[filtered_df['eqp_id'] == equipment_id]
            if chamber_id != 'All':
                filtered_df = filtered_df[filtered_df['chamber_id'] == chamber_id]
            if recipe != 'All':
                filtered_df = filtered_df[filtered_df['recipe'] == recipe]
            
            if len(filtered_df) == 0:
                # 빈 차트들 반환
                empty_fig = go.Figure()
                empty_fig.add_annotation(text="데이터가 없습니다", 
                                       xref="paper", yref="paper",
                                       x=0.5, y=0.5, showarrow=False)
                return html.Div("선택된 조건에 해당하는 데이터가 없습니다."), empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
            
            # 1. 통계 패널 (이상치 정보 + heuristic 회귀분석 정보 포함)
            temp_anomaly_count = len(filtered_df[filtered_df['TEMP'] > 450])
            pressure_anomaly_count = len(filtered_df[filtered_df['PRESSURE'] > 100])
            power_anomaly_count = len(filtered_df[filtered_df['RF_POWER'] > 300])
            load_anomaly_count = len(filtered_df[filtered_df['CHAMBER_LOAD'] > 800])
            total_anomalies = temp_anomaly_count + pressure_anomaly_count + power_anomaly_count + load_anomaly_count
            
            # heuristic.py 회귀분석 세그먼트 수 계산
            total_segments = 0
            linear_segments = 0
            features_for_analysis = ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']
            
            for feature in features_for_analysis:
                try:
                    results = analyze_timeseries_regression(filtered_df.reset_index(), feature)
                    total_segments += len(results)
                    linear_segments += sum(1 for r in results if r['linearity'] == 'linear')
                except Exception as e:
                    print(f"회귀분석 오류 ({feature}): {e}")
                    continue
            
            stats_panel = html.Div([
                html.Div([
                    html.H4(f"{len(filtered_df):,}", style={'margin': '0', 'color': '#007bff'}),
                    html.P("총 데이터 포인트", style={'margin': '0', 'font-size': '12px'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{filtered_df['TEMP'].mean():.1f}°C", style={'margin': '0', 'color': '#dc3545'}),
                    html.P("평균 온도", style={'margin': '0', 'font-size': '12px'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{filtered_df['PRESSURE'].mean():.1f}", style={'margin': '0', 'color': '#28a745'}),
                    html.P("평균 압력", style={'margin': '0', 'font-size': '12px'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{filtered_df['RF_POWER'].mean():.1f}", style={'margin': '0', 'color': '#ffc107'}),
                    html.P("평균 RF 파워", style={'margin': '0', 'font-size': '12px'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{temp_anomaly_count}", style={'margin': '0', 'color': '#e74c3c'}),
                    html.P("온도 이상치", style={'margin': '0', 'font-size': '12px'}),
                    html.P(f"({temp_anomaly_count/len(filtered_df)*100:.1f}%)", style={'margin': '0', 'font-size': '10px', 'color': '#666'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '10px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{power_anomaly_count}", style={'margin': '0', 'color': '#f39c12'}),
                    html.P("RF 파워 이상치", style={'margin': '0', 'font-size': '12px'}),
                    html.P(f"({power_anomaly_count/len(filtered_df)*100:.1f}%)", style={'margin': '0', 'font-size': '10px', 'color': '#666'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '10px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{pressure_anomaly_count}", style={'margin': '0', 'color': '#17a2b8'}),
                    html.P("압력 이상치", style={'margin': '0', 'font-size': '12px'}),
                    html.P(f"({pressure_anomaly_count/len(filtered_df)*100:.1f}%)", style={'margin': '0', 'font-size': '10px', 'color': '#666'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '10px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '13%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{total_anomalies}", style={'margin': '0', 'color': '#8e44ad'}),
                    html.P("총 이상치", style={'margin': '0', 'font-size': '12px'}),
                    html.P("🔴 원형 마커", style={'margin': '0', 'font-size': '10px', 'color': '#666'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '10px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '12%', 'display': 'inline-block', 'margin-right': '1%'}),
                
                html.Div([
                    html.H4(f"{total_segments}", style={'margin': '0', 'color': '#6c757d'}),
                    html.P("회귀 세그먼트", style={'margin': '0', 'font-size': '12px'}),
                    html.P(f"Window: {WINDOW_SIZE}", style={'margin': '0', 'font-size': '10px', 'color': '#666'})
                ], style={'text-align': 'center', 'background-color': 'white', 'padding': '10px', 
                         'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '12%', 'display': 'inline-block'})
            ])
            
            # 2. 시계열 차트 (heuristic.py 회귀분석 + 이상치 마커 포함)
            fig_ts = make_subplots(
                rows=2, cols=2,
                subplot_titles=('온도 (회귀분석 + 이상치)', '압력 (회귀분석 + 이상치)', 'RF 파워 (회귀분석 + 이상치)', '챔버 로드 (회귀분석 + 이상치)'),
                vertical_spacing=0.15
            )
            
            # 이상치 임계값 정의
            temp_threshold = 450
            pressure_threshold = 102.5
            power_threshold = 300
            load_threshold = 800
            
            # 이상치 데이터 식별
            temp_anomalies = filtered_df[filtered_df['TEMP'] > temp_threshold]
            pressure_anomalies = filtered_df[filtered_df['PRESSURE'] > pressure_threshold]
            power_anomalies = filtered_df[filtered_df['RF_POWER'] > power_threshold]
            load_anomalies = filtered_df[filtered_df['CHAMBER_LOAD'] > load_threshold]
            
            # 4개 변수에 대한 heuristic.py 회귀분석 적용
            features_analysis = [
                ('TEMP', '온도', 'red', 1, 1, temp_anomalies),
                ('PRESSURE', '압력', 'blue', 1, 2, pressure_anomalies),
                ('RF_POWER', 'RF 파워', 'green', 2, 1, power_anomalies),
                ('CHAMBER_LOAD', '챔버 로드', 'purple', 2, 2, load_anomalies)
            ]
            
            for feature, feature_name, color, row, col, anomalies in features_analysis:
                # 1. 원본 시계열 데이터
                fig_ts.add_trace(
                    go.Scatter(x=filtered_df['timestamp'], y=filtered_df[feature],
                              mode='lines', name=f'{feature_name}', 
                              line=dict(color=color, width=2)),
                    row=row, col=col
                )
                
                # 2. 이상치 원형 마커
                if len(anomalies) > 0:
                    # 유효한 색상 매핑
                    dark_color_map = {
                        'red': 'darkred',
                        'blue': 'darkblue', 
                        'green': 'darkgreen',
                        'purple': 'indigo'
                    }
                    dark_color = dark_color_map.get(color, 'black')
                    
                    fig_ts.add_trace(
                        go.Scatter(x=anomalies['timestamp'], y=anomalies[feature],
                                  mode='markers', name=f'{feature_name} 이상치',
                                  marker=dict(color=dark_color, size=8, symbol='circle-open', 
                                            line=dict(width=3, color=color)),
                                  showlegend=False,
                                  hovertemplate=f'<b>{feature_name} 이상치</b><br>시간: %{{x}}<br>{feature_name}: %{{y}}<extra></extra>'),
                        row=row, col=col
                    )
                
                # 3. heuristic.py 회귀분석 적용
                try:
                    regression_results = analyze_timeseries_regression(filtered_df.reset_index(), feature)
                except Exception as e:
                    print(f"회귀분석 오류 ({feature}): {e}")
                    regression_results = []
                
                # 세그먼트별 회귀선과 라벨 추가
                for i, result in enumerate(regression_results):
                    if result['alpha'] is not None:
                        # 세그먼트 구간 데이터
                        start_idx = result['segment'][0]
                        end_idx = result['segment'][1]
                        segment_data = filtered_df.iloc[start_idx:end_idx]
                        
                        if len(segment_data) >= 2:
                            # 회귀선 추가 (heuristic.py의 예측값 사용)
                            pred_values = result['pred']
                            if len(pred_values) == len(segment_data):
                                # 유효한 색상 매핑
                                dark_color_map = {
                                    'red': 'darkred',
                                    'blue': 'darkblue', 
                                    'green': 'darkgreen',
                                    'purple': 'indigo'
                                }
                                dark_color = dark_color_map.get(color, 'black')
                                
                                fig_ts.add_trace(
                                    go.Scatter(
                                        x=segment_data['timestamp'], 
                                        y=pred_values,
                                        mode='lines',
                                        name=f'{feature_name} 회귀선',
                                        line=dict(color=dark_color, width=3, dash='dot'),
                                        opacity=0.8,
                                        showlegend=False,
                                        hovertemplate=f'<b>{feature_name} 회귀선</b><br>' +
                                                    f'기울기: {result["slope"]:.4f}<br>' +
                                                    f'추세: {result["slope_sign"]}<br>' +
                                                    f'선형성: {result["linearity"]}<br>' +
                                                    f'P-value: {result["p_value"]:.4f}<br>' +
                                                    f'R²: {result["r_squared"]:.3f}<br>' +
                                                    f'Alpha: {result["alpha"]:.4f}<extra></extra>'
                                    ),
                                    row=row, col=col
                                )
                            
                            # 세그먼트 시작점에 추세 마커 추가 (heuristic.py와 동일한 라벨링)
                            start_time = result['start_time']
                            start_value = result['start_value']
                            
                            # 추세 마커 색상 및 기호 결정
                            if result['slope_sign'] == 'increase':
                                marker_symbol = 'triangle-up'
                                marker_color = 'darkgreen'
                                text_symbol = '↗'
                            elif result['slope_sign'] == 'decrease':
                                marker_symbol = 'triangle-down'
                                marker_color = 'darkred'
                                text_symbol = '↘'
                            else:  # stable
                                marker_symbol = 'diamond'
                                marker_color = 'orange'
                                text_symbol = '→'
                            
                            # 선형성에 따른 마커 크기 조정
                            marker_size = 12 if result['linearity'] == 'linear' else 8
                            
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=[start_time], 
                                    y=[start_value],
                                    mode='markers+text',
                                    name=f'{feature_name} 추세',
                                    marker=dict(
                                        color=marker_color, 
                                        size=marker_size, 
                                        symbol=marker_symbol,
                                        line=dict(width=2, color='white')
                                    ),
                                    text=[text_symbol],
                                    textposition="middle center",
                                    textfont=dict(size=10, color='white', family="Arial Black"),
                                    showlegend=False,
                                    hovertemplate=f'<b>{feature_name} 추세 분석</b><br>' +
                                                f'구간: {result["start_time"]} ~ {result["end_time"]}<br>' +
                                                f'길이: {result["segment_length"]}개<br>' +
                                                f'추세: {result["slope_sign"]}<br>' +
                                                f'선형성: {result["linearity"]}<br>' +
                                                f'기울기: {result["slope"]:.4f}<br>' +
                                                f'P-value: {result["p_value"]:.4f}<br>' +
                                                f'R²: {result["r_squared"]:.3f}<br>' +
                                                f'MSE 임계값: {MSE_THRESHOLD}<br>' +
                                                f'기울기 차이 임계값: {SLOPE_DIFF_THRESHOLD}<br>' +
                                                f'윈도우 크기: {WINDOW_SIZE}<extra></extra>'
                                ),
                                row=row, col=col
                            )
            
            fig_ts.update_layout(
                height=700, 
                title_text="시계열 데이터 (Heuristic 회귀분석 + 이상치 분석)", 
                showlegend=False
            )
            
            # heuristic.py 회귀분석 파라미터 및 마커 설명 추가
            fig_ts.add_annotation(
                text="<b>Heuristic 회귀분석 파라미터:</b><br>" +
                     f"🔧 윈도우 크기: {WINDOW_SIZE}<br>" +
                     f"📊 MSE 임계값: {MSE_THRESHOLD}<br>" +
                     f"📈 기울기 차이 임계값: {SLOPE_DIFF_THRESHOLD}<br>" +
                     "📋 P-value 임계값: 0.05<br>" +
                     "📊 R² 임계값: 0.3<br>" +
                     "📏 안정 기울기 임계값: ±0.05<br><br>" +
                     "<b>마커 설명:</b><br>" +
                     "🔴 이상치 (원형 마커)<br>" +
                     "↗ 증가 추세 (선형: 큰 삼각형)<br>" +
                     "↘ 감소 추세 (선형: 큰 삼각형)<br>" +
                     "→ 안정 추세 (다이아몬드)<br>" +
                     "⚪ 점선: Lasso 회귀선",
                xref="paper", yref="paper",
                x=0.02, y=0.98, showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="darkblue", borderwidth=2,
                font=dict(size=9)
            )
            
            # 3. 상관관계 히트맵
            features = ['TEMP', 'PRESSURE', 'RF_POWER', 'CHAMBER_LOAD']
            corr_matrix = filtered_df[features].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=features,
                y=features,
                colorscale='RdBu',
                zmid=0,
                text=np.around(corr_matrix.values, decimals=2),
                texttemplate="%{text}",
                textfont={"size": 12}
            ))
            fig_corr.update_layout(title="변수 간 상관관계", height=400)
            
            # 4. 이상치 탐지 (원형 마커 사용)
            temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
            power_anomalies = filtered_df[filtered_df['RF_POWER'] > 300]
            pressure_anomalies = filtered_df[filtered_df['PRESSURE'] > 100]
            combined_anomalies = filtered_df[
                (filtered_df['TEMP'] > 450) & (filtered_df['RF_POWER'] > 300)
            ]
            normal_data = filtered_df[
                (filtered_df['TEMP'] <= 450) & (filtered_df['RF_POWER'] <= 300)
            ]
            
            fig_anomaly = go.Figure()
            
            # 정상 데이터
            fig_anomaly.add_trace(
                go.Scatter(x=normal_data['TEMP'], y=normal_data['RF_POWER'],
                          mode='markers', name='정상',
                          marker=dict(color='lightblue', size=4, opacity=0.6),
                          hovertemplate='<b>정상 데이터</b><br>온도: %{x}°C<br>RF 파워: %{y}<extra></extra>')
            )
            
            # 온도 이상치 (원형 마커)
            if len(temp_anomalies) > 0:
                fig_anomaly.add_trace(
                    go.Scatter(x=temp_anomalies['TEMP'], y=temp_anomalies['RF_POWER'],
                              mode='markers', name=f'온도 이상치 ({len(temp_anomalies)}개)',
                              marker=dict(color='red', size=10, symbol='circle-open', 
                                        line=dict(width=3, color='darkred')),
                              hovertemplate='<b>온도 이상치</b><br>온도: %{x}°C<br>RF 파워: %{y}<br>임계값: 450°C 초과<extra></extra>')
                )
            
            # RF 파워 이상치 (원형 마커)
            if len(power_anomalies) > 0:
                fig_anomaly.add_trace(
                    go.Scatter(x=power_anomalies['TEMP'], y=power_anomalies['RF_POWER'],
                              mode='markers', name=f'RF 파워 이상치 ({len(power_anomalies)}개)',
                              marker=dict(color='orange', size=10, symbol='circle-open',
                                        line=dict(width=3, color='darkorange')),
                              hovertemplate='<b>RF 파워 이상치</b><br>온도: %{x}°C<br>RF 파워: %{y}<br>임계값: 300 초과<extra></extra>')
                )
            
            # 복합 이상치 (강조된 원형 마커)
            if len(combined_anomalies) > 0:
                fig_anomaly.add_trace(
                    go.Scatter(x=combined_anomalies['TEMP'], y=combined_anomalies['RF_POWER'],
                              mode='markers', name=f'복합 이상치 ({len(combined_anomalies)}개)',
                              marker=dict(color='darkred', size=15, symbol='circle-open',
                                        line=dict(width=4, color='yellow')),
                              hovertemplate='<b>⚠️ 위험: 복합 이상치</b><br>온도: %{x}°C<br>RF 파워: %{y}<br>두 값 모두 임계값 초과<extra></extra>')
                )
            
            # 임계값 선 추가
            fig_anomaly.add_hline(y=300, line_dash="dash", line_color="orange", 
                                 annotation_text="RF 파워 임계값 (300)", annotation_position="bottom right")
            fig_anomaly.add_vline(x=450, line_dash="dash", line_color="red",
                                 annotation_text="온도 임계값 (450°C)", annotation_position="top left")
            
            fig_anomaly.update_layout(
                title="이상치 탐지 분석 (온도 vs RF 파워) - 원형 마커",
                xaxis_title="온도 (°C)",
                yaxis_title="RF 파워",
                height=400,
                legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
            )
            
            # 5. 분포 차트 (이상치 마커 포함)
            fig_dist = make_subplots(
                rows=2, cols=2,
                subplot_titles=('온도 분포 (이상치 표시)', '압력 분포 (이상치 표시)', 'RF 파워 분포 (이상치 표시)', '챔버 로드 분포 (이상치 표시)')
            )
            
            # 이상치 임계값
            thresholds = {
                'TEMP': 450,
                'PRESSURE': 100, 
                'RF_POWER': 300,
                'CHAMBER_LOAD': 800
            }
            
            colors = ['red', 'blue', 'green', 'purple']
            
            for i, feature in enumerate(features):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                # 히스토그램
                fig_dist.add_trace(
                    go.Histogram(x=filtered_df[feature], nbinsx=30, name=feature,
                               marker_color=colors[i], opacity=0.7),
                    row=row, col=col
                )
                
                # 이상치 표시 (임계값 초과하는 값들)
                anomaly_data = filtered_df[filtered_df[feature] > thresholds[feature]]
                if len(anomaly_data) > 0:
                    # 이상치 값들의 y축 위치 계산 (히스토그램 위에 표시)
                    hist, bin_edges = np.histogram(filtered_df[feature], bins=30)
                    max_count = max(hist) if len(hist) > 0 else 1
                    
                    fig_dist.add_trace(
                        go.Scatter(x=anomaly_data[feature], 
                                 y=[max_count * 1.1] * len(anomaly_data),  # 히스토그램 위에 표시
                                 mode='markers', 
                                 name=f'{feature} 이상치',
                                 marker=dict(color='darkred', size=8, symbol='circle-open',
                                           line=dict(width=2, color=colors[i])),
                                 showlegend=False,
                                 hovertemplate=f'<b>{feature} 이상치</b><br>값: %{{x}}<br>임계값: {thresholds[feature]} 초과<extra></extra>'),
                        row=row, col=col
                    )
                    
                    # 임계값 선 추가
                    fig_dist.add_vline(x=thresholds[feature], line_dash="dash", 
                                     line_color="red", opacity=0.7,
                                     annotation_text=f"임계값: {thresholds[feature]}",
                                     annotation_position="top",
                                     row=row, col=col)
            
            fig_dist.update_layout(height=500, title_text="변수별 분포 (이상치 원형 마커 표시)", showlegend=False)
            
            # 6. 레시피별 비교 (이상치 마커 포함)
            fig_recipe = go.Figure()
            
            if recipe == 'All' and len(filtered_df['recipe'].unique()) > 1:
                # 레시피별 온도 박스 플롯
                for recipe_type in filtered_df['recipe'].unique():
                    recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
                    
                    # 박스 플롯
                    fig_recipe.add_trace(
                        go.Box(y=recipe_data['TEMP'], name=f'{recipe_type} 온도', 
                              boxpoints='outliers', marker_color='lightblue')
                    )
                    
                    # 이상치 원형 마커 추가
                    recipe_temp_anomalies = recipe_data[recipe_data['TEMP'] > 450]
                    if len(recipe_temp_anomalies) > 0:
                        fig_recipe.add_trace(
                            go.Scatter(x=[recipe_type] * len(recipe_temp_anomalies), 
                                     y=recipe_temp_anomalies['TEMP'],
                                     mode='markers', 
                                     name=f'{recipe_type} 이상치',
                                     marker=dict(color='red', size=10, symbol='circle-open',
                                               line=dict(width=3, color='darkred')),
                                     showlegend=False,
                                     hovertemplate=f'<b>{recipe_type} 온도 이상치</b><br>온도: %{{y}}°C<extra></extra>')
                        )
                
                # 임계값 선 추가
                fig_recipe.add_hline(y=450, line_dash="dash", line_color="red",
                                   annotation_text="온도 임계값 (450°C)")
            else:
                # 시간별 온도 변화
                filtered_df['hour'] = filtered_df['timestamp'].dt.hour
                hourly_avg = filtered_df.groupby('hour')['TEMP'].mean().reset_index()
                hourly_anomalies = filtered_df.groupby('hour').apply(
                    lambda x: len(x[x['TEMP'] > 450])
                ).reset_index(name='anomaly_count')
                
                # 시간별 평균 온도 라인
                fig_recipe.add_trace(
                    go.Scatter(x=hourly_avg['hour'], y=hourly_avg['TEMP'],
                              mode='lines+markers', name='시간별 평균 온도',
                              line=dict(color='blue', width=2),
                              marker=dict(size=6))
                )
                
                # 이상치가 있는 시간대 표시
                anomaly_hours = hourly_anomalies[hourly_anomalies['anomaly_count'] > 0]
                if len(anomaly_hours) > 0:
                    anomaly_temp_values = []
                    for hour in anomaly_hours['hour']:
                        temp_val = hourly_avg[hourly_avg['hour'] == hour]['TEMP'].iloc[0]
                        anomaly_temp_values.append(temp_val)
                    
                    fig_recipe.add_trace(
                        go.Scatter(x=anomaly_hours['hour'], y=anomaly_temp_values,
                                  mode='markers', name='이상치 발생 시간대',
                                  marker=dict(color='red', size=12, symbol='circle-open',
                                            line=dict(width=3, color='darkred')),
                                  hovertemplate='<b>이상치 발생 시간대</b><br>시간: %{x}시<br>평균 온도: %{y}°C<br>이상치 수: %{text}<extra></extra>',
                                  text=anomaly_hours['anomaly_count'])
                    )
                
                # 임계값 선 추가
                fig_recipe.add_hline(y=450, line_dash="dash", line_color="red",
                                   annotation_text="온도 임계값 (450°C)")
            
            fig_recipe.update_layout(
                title="레시피별 온도 비교 (이상치 원형 마커)" if recipe == 'All' else "시간별 온도 변화 (이상치 원형 마커)",
                height=400
            )
            
            return stats_panel, fig_ts, fig_corr, fig_anomaly, fig_dist, fig_recipe
            
        except Exception as e:
            # 전체 콜백 오류 처리
            print(f"대시보드 업데이트 오류: {e}")
            import traceback
            traceback.print_exc()
            
            # 오류 발생 시 빈 차트들과 오류 메시지 반환
            empty_fig = go.Figure()
            empty_fig.add_annotation(text=f"오류 발생: {str(e)}", 
                                   xref="paper", yref="paper",
                                   x=0.5, y=0.5, showarrow=False)
            error_message = html.Div([
                html.H4("오류 발생!", style={'color': 'red'}),
                html.P(f"상세 오류: {str(e)}")
            ])
            return error_message, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig
    
    return app

if __name__ == '__main__':
    print("=== 웹 대시보드 시작 ===")
    print("브라우저에서 http://127.0.0.1:8050/ 으로 접속하세요.")
    print("종료하려면 Ctrl+C를 누르세요.")
    
    try:
        app = create_web_dashboard()
        app.run(debug=True, host='127.0.0.1', port=8050)
    except Exception as e:
        print(f"❌ 웹 대시보드 실행 실패: {e}")
        print("필요한 패키지 설치: pip install dash plotly pandas numpy")
