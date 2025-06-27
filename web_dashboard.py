import dash
from dash import dcc, html, Input, Output, callback
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

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
        
        # 1. 통계 패널
        stats_panel = html.Div([
            html.Div([
                html.H4(f"{len(filtered_df):,}", style={'margin': '0', 'color': '#007bff'}),
                html.P("총 데이터 포인트", style={'margin': '0'})
            ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                     'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '18%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.H4(f"{filtered_df['TEMP'].mean():.1f}°C", style={'margin': '0', 'color': '#dc3545'}),
                html.P("평균 온도", style={'margin': '0'})
            ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                     'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '18%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.H4(f"{filtered_df['PRESSURE'].mean():.2f}", style={'margin': '0', 'color': '#28a745'}),
                html.P("평균 압력", style={'margin': '0'})
            ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                     'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '18%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.H4(f"{filtered_df['RF_POWER'].mean():.1f}", style={'margin': '0', 'color': '#ffc107'}),
                html.P("평균 RF 파워", style={'margin': '0'})
            ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                     'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '18%', 'display': 'inline-block', 'margin-right': '2%'}),
            
            html.Div([
                html.H4(f"{len(filtered_df[filtered_df['TEMP'] > 450])}", style={'margin': '0', 'color': '#e74c3c'}),
                html.P("온도 이상치", style={'margin': '0'})
            ], style={'text-align': 'center', 'background-color': 'white', 'padding': '15px', 
                     'border-radius': '10px', 'box-shadow': '0 2px 4px rgba(0,0,0,0.1)', 'width': '18%', 'display': 'inline-block'})
        ])
        
        # 2. 시계열 차트
        fig_ts = make_subplots(
            rows=2, cols=2,
            subplot_titles=('온도', '압력', 'RF 파워', '챔버 로드'),
            vertical_spacing=0.1
        )
        
        # 온도
        fig_ts.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['TEMP'],
                      mode='lines', name='온도', line=dict(color='red')),
            row=1, col=1
        )
        
        # 압력
        fig_ts.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['PRESSURE'],
                      mode='lines', name='압력', line=dict(color='blue')),
            row=1, col=2
        )
        
        # RF 파워
        fig_ts.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['RF_POWER'],
                      mode='lines', name='RF 파워', line=dict(color='green')),
            row=2, col=1
        )
        
        # 챔버 로드
        fig_ts.add_trace(
            go.Scatter(x=filtered_df['timestamp'], y=filtered_df['CHAMBER_LOAD'],
                      mode='lines', name='챔버 로드', line=dict(color='purple')),
            row=2, col=2
        )
        
        fig_ts.update_layout(height=600, title_text="시계열 데이터", showlegend=False)
        
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
        
        # 4. 이상치 탐지
        temp_anomalies = filtered_df[filtered_df['TEMP'] > 450]
        power_anomalies = filtered_df[filtered_df['RF_POWER'] > 300]
        normal_data = filtered_df[
            (filtered_df['TEMP'] <= 450) & (filtered_df['RF_POWER'] <= 300)
        ]
        
        fig_anomaly = go.Figure()
        
        # 정상 데이터
        fig_anomaly.add_trace(
            go.Scatter(x=normal_data['TEMP'], y=normal_data['RF_POWER'],
                      mode='markers', name='정상',
                      marker=dict(color='blue', size=4, opacity=0.6))
        )
        
        # 온도 이상치
        if len(temp_anomalies) > 0:
            fig_anomaly.add_trace(
                go.Scatter(x=temp_anomalies['TEMP'], y=temp_anomalies['RF_POWER'],
                          mode='markers', name='온도 이상',
                          marker=dict(color='red', size=6, symbol='x'))
            )
        
        # RF 파워 이상치
        if len(power_anomalies) > 0:
            fig_anomaly.add_trace(
                go.Scatter(x=power_anomalies['TEMP'], y=power_anomalies['RF_POWER'],
                          mode='markers', name='RF 파워 이상',
                          marker=dict(color='orange', size=6, symbol='diamond'))
            )
        
        fig_anomaly.update_layout(
            title="이상치 탐지 (온도 vs RF 파워)",
            xaxis_title="온도 (°C)",
            yaxis_title="RF 파워",
            height=400
        )
        
        # 5. 분포 차트
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=('온도 분포', '압력 분포', 'RF 파워 분포', '챔버 로드 분포')
        )
        
        for i, feature in enumerate(features):
            row = (i // 2) + 1
            col = (i % 2) + 1
            fig_dist.add_trace(
                go.Histogram(x=filtered_df[feature], nbinsx=30, name=feature),
                row=row, col=col
            )
        
        fig_dist.update_layout(height=500, title_text="변수별 분포", showlegend=False)
        
        # 6. 레시피별 비교
        fig_recipe = go.Figure()
        
        if recipe == 'All' and len(filtered_df['recipe'].unique()) > 1:
            for recipe_type in filtered_df['recipe'].unique():
                recipe_data = filtered_df[filtered_df['recipe'] == recipe_type]
                fig_recipe.add_trace(
                    go.Box(y=recipe_data['TEMP'], name=f'{recipe_type} 온도', 
                          boxpoints='outliers')
                )
        else:
            # 시간별 온도 변화
            filtered_df['hour'] = filtered_df['timestamp'].dt.hour
            hourly_avg = filtered_df.groupby('hour')['TEMP'].mean().reset_index()
            fig_recipe.add_trace(
                go.Scatter(x=hourly_avg['hour'], y=hourly_avg['TEMP'],
                          mode='lines+markers', name='시간별 평균 온도')
            )
        
        fig_recipe.update_layout(
            title="레시피별 온도 비교" if recipe == 'All' else "시간별 온도 변화",
            height=400
        )
        
        return stats_panel, fig_ts, fig_corr, fig_anomaly, fig_dist, fig_recipe
    
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
