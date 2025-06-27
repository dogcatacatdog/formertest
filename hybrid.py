import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy import stats
from scipy.stats import jarque_bera, shapiro
import warnings
warnings.filterwarnings('ignore')

# 1. Adaptive Series Decomposition (kernel_size 자동 조정)
class AdaptiveSeriesDecomposition(nn.Module):
    def __init__(self, min_kernel=5, max_kernel=51):
        super().__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel

    def forward(self, x):
        # dominant 주기 추정 (FFT peak)
        B, L, D = x.shape
        fft = torch.fft.rfft(x, dim=1)
        spectrum = fft.abs().mean(dim=(0,2))
        peak_idx = spectrum[1:].argmax() + 1
        period = max(self.min_kernel, min(self.max_kernel, int(L / peak_idx))) if peak_idx > 0 else self.min_kernel
        # kernel_size 동적 결정 - 홀수로 보정
        if period % 2 == 0:
            period += 1
        pad = period // 2
        trend = F.avg_pool1d(x.permute(0,2,1), kernel_size=period, stride=1, padding=pad, count_include_pad=False).permute(0,2,1)
        # 크기 일치 보정
        if trend.size(1) != L:
            trend = trend[:, :L, :]
        seasonal = x - trend
        return seasonal, trend

# 2. ProbSparse Attention with dynamic top-k
class DynamicProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, min_factor=3, max_factor=10):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def get_sparsity_score(self, Q):
        l2 = Q.norm(dim=-1)
        prob = F.softmax(Q, dim=-1)
        entropy = -torch.sum(prob * torch.log(prob + 1e-9), dim=-1)
        var = torch.var(Q, dim=-1)
        return 0.5 * l2 + 0.3 * entropy + 0.2 * var

    def forward(self, x):
        B, L, D = x.shape
        Q = self.w_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        # 동적 top-k: 분산이 높을수록 더 많은 쿼리 사용
        q_scores = self.get_sparsity_score(Q)
        q_std = q_scores.std(dim=2, keepdim=True)
        factor = torch.clamp((q_std / (q_scores.mean(dim=2, keepdim=True)+1e-6) * self.max_factor).round().int(), self.min_factor, self.max_factor)
        k_list = factor.squeeze(-1).tolist()
        # 배치별 top-k 적용
        outputs = []
        for b in range(B):
            u = int(k_list[b][0])
            _, top_idx = torch.topk(q_scores[b], k=min(u, L), dim=-1)
            Q_sparse = torch.gather(Q[b], 1, top_idx.unsqueeze(-1).expand(-1, -1, self.d_k))
            scores = torch.matmul(Q_sparse, K[b].transpose(-2, -1)) / math.sqrt(self.d_k)
            attn_weights = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_weights, V[b])
            full_out = torch.zeros(self.n_heads, L, self.d_k, device=x.device)
            for h in range(self.n_heads):
                full_out[h, top_idx[h]] = out[h]
            outputs.append(full_out.transpose(0,1).reshape(L, D))
        output = torch.stack(outputs, dim=0)
        return self.w_o(output), None

# 3. Patch Embedding (PatchTST 스타일, 옵션)
class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, d_model, patch_len=16):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(input_dim * patch_len, d_model)

    def forward(self, x):
        B, L, D = x.shape
        pad = (self.patch_len - (L % self.patch_len)) % self.patch_len
        if pad > 0:
            x = F.pad(x, (0,0,0,pad))
        x = x.unfold(1, self.patch_len, self.patch_len) # [B, num_patch, patch_len, D]
        x = x.contiguous().view(B, -1, self.patch_len * D)
        return self.proj(x)

# 4. Dynamic Threshold (Percentile+EWMA, 자동화)
class DynamicThreshold:
    def __init__(self, ewma_alpha=0.2, min_percentile=90, max_percentile=99):
        self.ewma = None
        self.alpha = ewma_alpha
        self.min_percentile = min_percentile
        self.max_percentile = max_percentile
        self.history = []
    def update(self, scores):
        scores = scores.detach().cpu()
        self.history.append(scores)
        if len(self.history) > 50:
            self.history = self.history[-50:]
        all_scores = torch.cat(self.history)
        # 자동화: 분산 높으면 percentile 높임(오탐 감소), 낮으면 낮춤(민감도↑)
        std = all_scores.std().item()
        percentile = int(self.min_percentile + (self.max_percentile-self.min_percentile)*min(std/2,1))
        thres = torch.quantile(all_scores, percentile / 100.0).item()
        self.ewma = thres if self.ewma is None else self.alpha * thres + (1 - self.alpha) * self.ewma
        return self.ewma

# 5. Hybrid Model (Encoder-only, 파라미터 공유, 정규화 자동화)
class InformerAutoformerHybrid(nn.Module):
    def __init__(self, input_dim=1, d_model=64, n_heads=4, d_ff=128, num_layers=2, out_len=24, use_patch=False, patch_len=16, share_params=True):
        super().__init__()
        self.use_patch = use_patch
        if use_patch:
            self.embed = PatchEmbedding(input_dim, d_model, patch_len)
        else:
            self.embed = nn.Linear(input_dim, d_model)
        # 파라미터 공유
        encoder_layer = nn.ModuleDict({
            'attn': DynamicProbSparseAttention(d_model, n_heads),
            'ffn': nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ),
            'norm1': nn.LayerNorm(d_model),
            'norm2': nn.LayerNorm(d_model)
        })
        self.layers = nn.ModuleList([encoder_layer if share_params else encoder_layer.deepcopy() for _ in range(num_layers)])
        self.decomp = AdaptiveSeriesDecomposition()
        self.out_proj = nn.Linear(d_model, 1)
        self.out_len = out_len

    def forward(self, x, y_true=None):
        if self.use_patch:
            x = self.embed(x)
        else:
            x = self.embed(x)
        attn_maps = []
        for layer in self.layers:
            attn_out, attn_map = layer['attn'](x)
            attn_maps.append(attn_map)
            x = layer['norm1'](x + attn_out)
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        seasonal, trend = self.decomp(x)
        combined = seasonal + trend
        # 출력 길이 자동 조정
        if combined.size(1) >= self.out_len:
            out_seq = combined[:, -self.out_len:, :]
        else:
            pad = combined[:, -1:, :].repeat(1, self.out_len - combined.size(1), 1)
            out_seq = torch.cat([combined, pad], dim=1)
        pred = self.out_proj(out_seq).squeeze(-1)
        # 이상감지: 예측 오차 기반 self-supervised scoring
        if y_true is not None:
            error = torch.abs(pred - y_true)
            anomaly_score = error.mean(dim=1)
        else:
            anomaly_score = None
        return {'forecast': pred, 'anomaly_score': anomaly_score, 'attn_maps': attn_maps, 'seasonal': seasonal, 'trend': trend}

# 7. Regression Analysis Module (독립적 회귀분석)
class TrendRegressionAnalyzer:
    def __init__(self, window_size=50, lasso_alpha=0.01, p_threshold=0.005):
        """
        독립적 회귀분석 모듈
        
        Args:
            window_size: 회귀분석용 sliding window 크기
            lasso_alpha: Lasso 회귀 정규화 파라미터
            p_threshold: 선형/비선형성 판정 p-value 임계값
        """
        self.window_size = window_size
        self.lasso_alpha = lasso_alpha
        self.p_threshold = p_threshold
        
        # 회귀 모델들
        self.linear_reg = LinearRegression()
        self.lasso_reg = Lasso(alpha=lasso_alpha, max_iter=1000)
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
        # 결과 저장
        self.trend_results = []
        self.linearity_results = []
        
    def sliding_window_analysis(self, data):
        """
        슬라이딩 윈도우 기반 회귀분석
        
        Args:
            data: 시계열 데이터 (numpy array or tensor)
            
        Returns:
            dict: 회귀분석 결과
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if len(data.shape) > 1:
            data = data.flatten()
            
        n = len(data)
        if n < self.window_size:
            return self._analyze_single_window(data, 0)
            
        results = []
        
        # 슬라이딩 윈도우 적용
        for i in range(n - self.window_size + 1):
            window_data = data[i:i + self.window_size]
            window_result = self._analyze_single_window(window_data, i)
            results.append(window_result)
            
        return self._aggregate_results(results)
        
    def _analyze_single_window(self, window_data, start_idx):
        """
        단일 윈도우 회귀분석
        """
        n = len(window_data)
        X = np.arange(n).reshape(-1, 1)
        y = window_data
        
        # 1. 선형 회귀
        self.linear_reg.fit(X, y)
        linear_pred = self.linear_reg.predict(X)
        linear_r2 = r2_score(y, linear_pred)
        linear_slope = self.linear_reg.coef_[0]
        
        # 2. Lasso 회귀
        self.lasso_reg.fit(X, y)
        lasso_pred = self.lasso_reg.predict(X)
        lasso_r2 = r2_score(y, lasso_pred)
        lasso_slope = self.lasso_reg.coef_[0]
        
        # 3. 다항 회귀 (2차)
        X_poly = self.poly_features.fit_transform(X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        poly_pred = poly_reg.predict(X_poly)
        poly_r2 = r2_score(y, poly_pred)
        
        # 4. 선형성 검정 (F-test)
        linear_residuals = y - linear_pred
        poly_residuals = y - poly_pred
        
        # F-통계량 계산
        rss_linear = np.sum(linear_residuals**2)
        rss_poly = np.sum(poly_residuals**2)
        
        if rss_poly > 0:
            f_stat = ((rss_linear - rss_poly) / 1) / (rss_poly / (n - 3))
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
        else:
            f_stat = 0
            p_value = 1.0
            
        # 5. 추세 판정
        trend_direction = self._classify_trend(linear_slope, p_value)
        
        # 6. 정규성 검정 (Shapiro-Wilk)
        try:
            shapiro_stat, shapiro_p = shapiro(linear_residuals)
        except:
            shapiro_stat, shapiro_p = 0, 1
            
        # 7. 자기상관 검정 (Durbin-Watson)
        dw_stat = self._durbin_watson(linear_residuals)
        
        return {
            'start_idx': start_idx,
            'window_size': n,
            'linear_r2': linear_r2,
            'linear_slope': linear_slope,
            'lasso_r2': lasso_r2,
            'lasso_slope': lasso_slope,
            'poly_r2': poly_r2,
            'f_statistic': f_stat,
            'p_value': p_value,
            'is_linear': p_value > self.p_threshold,
            'trend_direction': trend_direction,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'durbin_watson': dw_stat,
            'residuals_std': np.std(linear_residuals)
        }
        
    def _classify_trend(self, slope, p_value):
        """
        추세 분류
        """
        if p_value <= self.p_threshold:  # 비선형
            return 'nonlinear'
        elif abs(slope) < 1e-6:  # 기울기가 거의 0
            return 'flat'
        elif slope > 0:
            if slope > 0.1:
                return 'strong_increasing'
            else:
                return 'weak_increasing'
        else:
            if slope < -0.1:
                return 'strong_decreasing'
            else:
                return 'weak_decreasing'
                
    def _durbin_watson(self, residuals):
        """
        Durbin-Watson 통계량 계산
        """
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
        
    def _aggregate_results(self, results):
        """
        윈도우별 결과 집계
        """
        if not results:
            return {}
            
        # 평균 통계
        avg_linear_r2 = np.mean([r['linear_r2'] for r in results])
        avg_lasso_r2 = np.mean([r['lasso_r2'] for r in results])
        avg_poly_r2 = np.mean([r['poly_r2'] for r in results])
        
        # 추세 분포
        trends = [r['trend_direction'] for r in results]
        trend_counts = {trend: trends.count(trend) for trend in set(trends)}
        dominant_trend = max(trend_counts, key=trend_counts.get)
        
        # 선형성 비율
        linear_ratio = np.mean([r['is_linear'] for r in results])
        
        # 평균 기울기
        avg_slope = np.mean([r['linear_slope'] for r in results])
        slope_std = np.std([r['linear_slope'] for r in results])
        
        return {
            'summary': {
                'avg_linear_r2': avg_linear_r2,
                'avg_lasso_r2': avg_lasso_r2,
                'avg_poly_r2': avg_poly_r2,
                'dominant_trend': dominant_trend,
                'trend_distribution': trend_counts,
                'linear_ratio': linear_ratio,
                'avg_slope': avg_slope,
                'slope_std': slope_std,
                'total_windows': len(results)
            },
            'detailed_results': results
        }
        
    def analyze_trend_stability(self, data):
        """
        추세 안정성 분석
        """
        results = self.sliding_window_analysis(data)
        
        if 'detailed_results' not in results:
            return {}
            
        slopes = [r['linear_slope'] for r in results['detailed_results']]
        r2_scores = [r['linear_r2'] for r in results['detailed_results']]
        
        # 기울기 변화율
        slope_changes = np.diff(slopes)
        slope_volatility = np.std(slope_changes)
        
        # R² 안정성
        r2_stability = 1 - (np.std(r2_scores) / (np.mean(r2_scores) + 1e-6))
        
        # 추세 변화점 탐지
        trend_changes = []
        for i in range(1, len(slopes)):
            if np.sign(slopes[i]) != np.sign(slopes[i-1]) and abs(slopes[i]) > 1e-6:
                trend_changes.append(i)
                
        return {
            'slope_volatility': slope_volatility,
            'r2_stability': r2_stability,
            'trend_change_points': trend_changes,
            'num_trend_changes': len(trend_changes),
            'slope_range': (min(slopes), max(slopes)),
            'slope_trend': 'increasing' if slopes[-1] > slopes[0] else 'decreasing'
        }

# 8. Enhanced Hybrid Model with Regression Analysis
class EnhancedInformerAutoformerHybrid(InformerAutoformerHybrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_analyzer = TrendRegressionAnalyzer()
        
    def forward_with_regression(self, x, y_true=None):
        """
        회귀분석이 포함된 forward pass
        """
        # 기본 forward
        basic_output = self.forward(x, y_true)
        
        # 추출된 trend에 대한 회귀분석
        trend_data = basic_output['trend']  # [B, L, D]
        
        regression_results = []
        for b in range(trend_data.size(0)):
            for d in range(trend_data.size(2)):
                trend_series = trend_data[b, :, d]
                reg_result = self.trend_analyzer.sliding_window_analysis(trend_series)
                regression_results.append(reg_result)
                
        # 결과 통합
        enhanced_output = basic_output.copy()
        enhanced_output['regression_analysis'] = regression_results
        enhanced_output['trend_stability'] = [
            self.trend_analyzer.analyze_trend_stability(trend_data[b, :, d])
            for b in range(trend_data.size(0))
            for d in range(trend_data.size(2))
        ]
        
        return enhanced_output

# 6. Example usage with Regression Analysis
if __name__ == "__main__":
    # 기본 모델 테스트
    model = InformerAutoformerHybrid(input_dim=3, d_model=64, out_len=24, use_patch=True, patch_len=12, share_params=True)
    
    # Enhanced 모델 테스트 (회귀분석 포함)
    enhanced_model = EnhancedInformerAutoformerHybrid(input_dim=3, d_model=64, out_len=24, use_patch=True, patch_len=12, share_params=True)
    
    optimizer = torch.optim.Adam(enhanced_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_x = torch.randn(100, 96, 3)
    train_y = torch.randn(100, 24)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y), batch_size=8, shuffle=True
    )
    thres = DynamicThreshold()
    
    # 훈련
    for epoch in range(3):
        enhanced_model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = enhanced_model(batch_x, batch_y)
            loss = criterion(out['forecast'], batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")
    
    # 테스트 및 회귀분석
    enhanced_model.eval()
    test_x = torch.randn(10, 96, 3)
    test_y = torch.randn(10, 24)
    
    with torch.no_grad():
        # 기본 출력
        out = enhanced_model(test_x, test_y)
        anomaly_score = out['anomaly_score']
        threshold = thres.update(anomaly_score)
        anomaly_mask = (anomaly_score > threshold)
        
        print(f"=== 기본 이상탐지 결과 ===")
        print(f"Anomaly Score: {anomaly_score}")
        print(f"Dynamic Threshold: {threshold:.4f}")
        print(f"Anomaly Mask: {anomaly_mask}")
        
        # 회귀분석 포함 출력
        enhanced_out = enhanced_model.forward_with_regression(test_x, test_y)
        
        print(f"\n=== 회귀분석 결과 ===")
        if enhanced_out['regression_analysis']:
            for i, reg_result in enumerate(enhanced_out['regression_analysis'][:3]):  # 처음 3개만 출력
                if 'summary' in reg_result:
                    summary = reg_result['summary']
                    print(f"Sample {i+1}:")
                    print(f"  평균 선형 R²: {summary['avg_linear_r2']:.4f}")
                    print(f"  평균 Lasso R²: {summary['avg_lasso_r2']:.4f}")
                    print(f"  지배적 추세: {summary['dominant_trend']}")
                    print(f"  선형성 비율: {summary['linear_ratio']:.4f}")
                    print(f"  평균 기울기: {summary['avg_slope']:.6f}")
                    print(f"  추세 분포: {summary['trend_distribution']}")
        
        print(f"\n=== 추세 안정성 분석 ===")
        if enhanced_out['trend_stability']:
            for i, stability in enumerate(enhanced_out['trend_stability'][:3]):  # 처음 3개만 출력
                if stability:
                    print(f"Sample {i+1}:")
                    print(f"  기울기 변동성: {stability['slope_volatility']:.6f}")
                    print(f"  R² 안정성: {stability['r2_stability']:.4f}")
                    print(f"  추세 변화점 수: {stability['num_trend_changes']}")
                    print(f"  기울기 범위: {stability['slope_range']}")
                    print(f"  전체 기울기 추세: {stability['slope_trend']}")
    
    # 독립적 회귀분석 테스트
    print(f"\n=== 독립적 회귀분석 테스트 ===")
    trend_analyzer = TrendRegressionAnalyzer(window_size=30, lasso_alpha=0.01, p_threshold=0.005)
    
    # 샘플 시계열 데이터 생성 (선형 + 노이즈)
    time_steps = np.arange(100)
    sample_trend = 0.5 * time_steps + 10 + np.random.normal(0, 2, 100)
    
    # 회귀분석 수행
    reg_result = trend_analyzer.sliding_window_analysis(sample_trend)
    
    if 'summary' in reg_result:
        summary = reg_result['summary']
        print(f"샘플 데이터 회귀분석:")
        print(f"  평균 선형 R²: {summary['avg_linear_r2']:.4f}")
        print(f"  평균 Lasso R²: {summary['avg_lasso_r2']:.4f}")
        print(f"  지배적 추세: {summary['dominant_trend']}")
        print(f"  선형성 비율: {summary['linear_ratio']:.4f}")
        print(f"  평균 기울기: {summary['avg_slope']:.6f}")
        print(f"  기울기 표준편차: {summary['slope_std']:.6f}")
        print(f"  총 윈도우 수: {summary['total_windows']}")
    
    # 추세 안정성 분석
    stability = trend_analyzer.analyze_trend_stability(sample_trend)
    print(f"\n추세 안정성:")
    print(f"  기울기 변동성: {stability['slope_volatility']:.6f}")
    print(f"  R² 안정성: {stability['r2_stability']:.4f}")
    print(f"  추세 변화점 수: {stability['num_trend_changes']}")
    print(f"  기울기 범위: {stability['slope_range']}")
    print(f"  전체 기울기 추세: {stability['slope_trend']}")

# 7. Regression Analysis Module (독립적 회귀분석)
class TrendRegressionAnalyzer:
    def __init__(self, window_size=50, lasso_alpha=0.01, p_threshold=0.005):
        """
        독립적 회귀분석 모듈
        
        Args:
            window_size: 회귀분석용 sliding window 크기
            lasso_alpha: Lasso 회귀 정규화 파라미터
            p_threshold: 선형/비선형성 판정 p-value 임계값
        """
        self.window_size = window_size
        self.lasso_alpha = lasso_alpha
        self.p_threshold = p_threshold
        
        # 회귀 모델들
        self.linear_reg = LinearRegression()
        self.lasso_reg = Lasso(alpha=lasso_alpha, max_iter=1000)
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
        # 결과 저장
        self.trend_results = []
        self.linearity_results = []
        
    def sliding_window_analysis(self, data):
        """
        슬라이딩 윈도우 기반 회귀분석
        
        Args:
            data: 시계열 데이터 (numpy array or tensor)
            
        Returns:
            dict: 회귀분석 결과
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if len(data.shape) > 1:
            data = data.flatten()
            
        n = len(data)
        if n < self.window_size:
            return self._analyze_single_window(data, 0)
            
        results = []
        
        # 슬라이딩 윈도우 적용
        for i in range(n - self.window_size + 1):
            window_data = data[i:i + self.window_size]
            window_result = self._analyze_single_window(window_data, i)
            results.append(window_result)
            
        return self._aggregate_results(results)
        
    def _analyze_single_window(self, window_data, start_idx):
        """
        단일 윈도우 회귀분석
        """
        n = len(window_data)
        X = np.arange(n).reshape(-1, 1)
        y = window_data
        
        # 1. 선형 회귀
        self.linear_reg.fit(X, y)
        linear_pred = self.linear_reg.predict(X)
        linear_r2 = r2_score(y, linear_pred)
        linear_slope = self.linear_reg.coef_[0]
        
        # 2. Lasso 회귀
        self.lasso_reg.fit(X, y)
        lasso_pred = self.lasso_reg.predict(X)
        lasso_r2 = r2_score(y, lasso_pred)
        lasso_slope = self.lasso_reg.coef_[0]
        
        # 3. 다항 회귀 (2차)
        X_poly = self.poly_features.fit_transform(X)
        poly_reg = LinearRegression()
        poly_reg.fit(X_poly, y)
        poly_pred = poly_reg.predict(X_poly)
        poly_r2 = r2_score(y, poly_pred)
        
        # 4. 선형성 검정 (F-test)
        linear_residuals = y - linear_pred
        poly_residuals = y - poly_pred
        
        # F-통계량 계산
        rss_linear = np.sum(linear_residuals**2)
        rss_poly = np.sum(poly_residuals**2)
        
        if rss_poly > 0:
            f_stat = ((rss_linear - rss_poly) / 1) / (rss_poly / (n - 3))
            p_value = 1 - stats.f.cdf(f_stat, 1, n - 3)
        else:
            f_stat = 0
            p_value = 1.0
            
        # 5. 추세 판정
        trend_direction = self._classify_trend(linear_slope, p_value)
        
        # 6. 정규성 검정 (Shapiro-Wilk)
        try:
            shapiro_stat, shapiro_p = shapiro(linear_residuals)
        except:
            shapiro_stat, shapiro_p = 0, 1
            
        # 7. 자기상관 검정 (Durbin-Watson)
        dw_stat = self._durbin_watson(linear_residuals)
        
        return {
            'start_idx': start_idx,
            'window_size': n,
            'linear_r2': linear_r2,
            'linear_slope': linear_slope,
            'lasso_r2': lasso_r2,
            'lasso_slope': lasso_slope,
            'poly_r2': poly_r2,
            'f_statistic': f_stat,
            'p_value': p_value,
            'is_linear': p_value > self.p_threshold,
            'trend_direction': trend_direction,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'durbin_watson': dw_stat,
            'residuals_std': np.std(linear_residuals)
        }
        
    def _classify_trend(self, slope, p_value):
        """
        추세 분류
        """
        if p_value <= self.p_threshold:  # 비선형
            return 'nonlinear'
        elif abs(slope) < 1e-6:  # 기울기가 거의 0
            return 'flat'
        elif slope > 0:
            if slope > 0.1:
                return 'strong_increasing'
            else:
                return 'weak_increasing'
        else:
            if slope < -0.1:
                return 'strong_decreasing'
            else:
                return 'weak_decreasing'
                
    def _durbin_watson(self, residuals):
        """
        Durbin-Watson 통계량 계산
        """
        diff = np.diff(residuals)
        return np.sum(diff**2) / np.sum(residuals**2)
        
    def _aggregate_results(self, results):
        """
        윈도우별 결과 집계
        """
        if not results:
            return {}
            
        # 평균 통계
        avg_linear_r2 = np.mean([r['linear_r2'] for r in results])
        avg_lasso_r2 = np.mean([r['lasso_r2'] for r in results])
        avg_poly_r2 = np.mean([r['poly_r2'] for r in results])
        
        # 추세 분포
        trends = [r['trend_direction'] for r in results]
        trend_counts = {trend: trends.count(trend) for trend in set(trends)}
        dominant_trend = max(trend_counts, key=trend_counts.get)
        
        # 선형성 비율
        linear_ratio = np.mean([r['is_linear'] for r in results])
        
        # 평균 기울기
        avg_slope = np.mean([r['linear_slope'] for r in results])
        slope_std = np.std([r['linear_slope'] for r in results])
        
        return {
            'summary': {
                'avg_linear_r2': avg_linear_r2,
                'avg_lasso_r2': avg_lasso_r2,
                'avg_poly_r2': avg_poly_r2,
                'dominant_trend': dominant_trend,
                'trend_distribution': trend_counts,
                'linear_ratio': linear_ratio,
                'avg_slope': avg_slope,
                'slope_std': slope_std,
                'total_windows': len(results)
            },
            'detailed_results': results
        }
        
    def analyze_trend_stability(self, data):
        """
        추세 안정성 분석
        """
        results = self.sliding_window_analysis(data)
        
        if 'detailed_results' not in results:
            return {}
            
        slopes = [r['linear_slope'] for r in results['detailed_results']]
        r2_scores = [r['linear_r2'] for r in results['detailed_results']]
        
        # 기울기 변화율
        slope_changes = np.diff(slopes)
        slope_volatility = np.std(slope_changes)
        
        # R² 안정성
        r2_stability = 1 - (np.std(r2_scores) / (np.mean(r2_scores) + 1e-6))
        
        # 추세 변화점 탐지
        trend_changes = []
        for i in range(1, len(slopes)):
            if np.sign(slopes[i]) != np.sign(slopes[i-1]) and abs(slopes[i]) > 1e-6:
                trend_changes.append(i)
                
        return {
            'slope_volatility': slope_volatility,
            'r2_stability': r2_stability,
            'trend_change_points': trend_changes,
            'num_trend_changes': len(trend_changes),
            'slope_range': (min(slopes), max(slopes)),
            'slope_trend': 'increasing' if slopes[-1] > slopes[0] else 'decreasing'
        }

# 8. Enhanced Hybrid Model with Regression Analysis
class EnhancedInformerAutoformerHybrid(InformerAutoformerHybrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trend_analyzer = TrendRegressionAnalyzer()
        
    def forward_with_regression(self, x, y_true=None):
        """
        회귀분석이 포함된 forward pass
        """
        # 기본 forward
        basic_output = self.forward(x, y_true)
        
        # 추출된 trend에 대한 회귀분석
        trend_data = basic_output['trend']  # [B, L, D]
        
        regression_results = []
        for b in range(trend_data.size(0)):
            for d in range(trend_data.size(2)):
                trend_series = trend_data[b, :, d]
                reg_result = self.trend_analyzer.sliding_window_analysis(trend_series)
                regression_results.append(reg_result)
                
        # 결과 통합
        enhanced_output = basic_output.copy()
        enhanced_output['regression_analysis'] = regression_results
        enhanced_output['trend_stability'] = [
            self.trend_analyzer.analyze_trend_stability(trend_data[b, :, d])
            for b in range(trend_data.size(0))
            for d in range(trend_data.size(2))
        ]
        
        return enhanced_output
