import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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

# 6. Example usage
if __name__ == "__main__":
    model = InformerAutoformerHybrid(input_dim=3, d_model=64, out_len=24, use_patch=True, patch_len=12, share_params=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    train_x = torch.randn(100, 96, 3)
    train_y = torch.randn(100, 24)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x, train_y), batch_size=8, shuffle=True
    )
    thres = DynamicThreshold()
    for epoch in range(3):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_x, batch_y)
            loss = criterion(out['forecast'], batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} done.")
    model.eval()
    test_x = torch.randn(10, 96, 3)
    test_y = torch.randn(10, 24)
    with torch.no_grad():
        out = model(test_x, test_y)
        anomaly_score = out['anomaly_score']
        threshold = thres.update(anomaly_score)
        anomaly_mask = (anomaly_score > threshold)
        print(f"Anomaly Score: {anomaly_score}")
        print(f"Dynamic Threshold: {threshold:.4f}")
        print(f"Anomaly Mask: {anomaly_mask}")
