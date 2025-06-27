import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

MSE_THRESHOLD = 0.1
SLOPE_DIFF_THRESHOLD = 0.25
WINDOW_SIZE = 30

def heuristic_alpha(X, y):
    y_std = np.std(y)
    x_range = X['x'].max() - X['x'].min()
    if x_range == 0:
        return 0.01
    scale = y_std / x_range
    return max(min(scale, 1.0), 0.01)

def evaluate_merge_condition(data, idx1, idx2):
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
    return [(i, i + WINDOW_SIZE) for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE)]

def bottom_up_merge(data, segments):
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
    df_seg = data.iloc[seg[0]:seg[1]]
    if df_seg.isnull().any().any() or len(df_seg) < 2:
        return {'segment': seg, 'alpha': None, 'slope': np.nan, 'slope_sign': 'N/A', 'linearity': 'N/A', 'pred': np.full(len(df_seg), np.nan)}
    alpha = heuristic_alpha(df_seg[['x']], df_seg['y'])
    model = Lasso(alpha=alpha).fit(df_seg[['x']], df_seg['y'])
    pred = model.predict(df_seg[['x']])
    slope = model.coef_[0]

    if abs(slope) <= 0.05:
        slope_sign = 'stable'
    elif slope > 0:
        slope_sign = 'increase'
    else:
        slope_sign = 'decrease'

    X_const = sm.add_constant(df_seg[['x']])
    ols = sm.OLS(df_seg['y'], X_const).fit()
    linearity = 'linear' if (ols.pvalues[1] <= 0.05 and ols.rsquared >= 0.3) else 'non-linear'

    return {'segment': seg, 'alpha': alpha, 'slope': slope, 'slope_sign': slope_sign, 'linearity': linearity, 'pred': pred}

def save_analysis_results(df, eqp_columns, save_path):
    for eqp_col in eqp_columns:
        data = pd.DataFrame({'x': df.index, 'y': df[eqp_col]})
        segments = generate_initial_segments(data)
        merged_segments = bottom_up_merge(data, segments)
        results = [process_segment_label(seg, data) for seg in merged_segments]

        prev_label = None
        prev_slope = None
        for result in results:
            seg = result['segment']
            linearity = result['linearity']
            slope_sign = result['slope_sign']
            idx = seg[0]
            if prev_label is None or prev_label != linearity:
                df.at[idx, f"{eqp_col}_result_linear"] = linearity
            if prev_slope is None or prev_slope != slope_sign:
                df.at[idx, f"{eqp_col}_result_slope"] = slope_sign
            prev_label = linearity
            prev_slope = slope_sign

    df.to_csv(save_path, index=False)
    return df

if __name__ == "__main__":
    df = pd.read_csv(r"C:\Users\WD\Desktop\work2\back-end\flask-vite-app\backend\analysis\eqp_regression_test_sample.csv")
    eqp_columns = ['eqp1', 'eqp2', 'eqp3', 'eqp4']
    for eqp in eqp_columns:
        df[f"{eqp}_result_linear"] = ""
        df[f"{eqp}_result_slope"] = ""
    save_path = r"C:\Users\WD\Desktop\work2\back-end\flask-vite-app\backend\analysis\result.csv"
    result_df = save_analysis_results(df.copy(), eqp_columns, save_path)
    print(f"분석 완료 → {save_path}")

    for eqp in eqp_columns:
        plt.figure(figsize=(14, 5))
        plt.plot(result_df['date'] if 'date' in result_df else result_df.index, result_df[eqp], marker='o', label=f"{eqp} value", color='black', linewidth=1)

        for idx, row in result_df.iterrows():
            label = row.get(f"{eqp}_result_linear", "")
            slope = row.get(f"{eqp}_result_slope", "")
            label_texts = []
            if pd.notna(label) and label not in ["", "N/A", "nan"]:
                label_texts.append(str(label))
            if pd.notna(slope) and slope not in ["", "N/A", "nan"]:
                label_texts.append(str(slope))
            if label_texts:
                plt.text(
                    row['date'] if 'date' in row else idx,
                    row[eqp],
                    "\n".join(label_texts),
                    color='red', fontsize=9, fontweight='bold', ha='center', va='bottom'
                )

        plt.title(f"{eqp} 분석 결과")
        plt.xlabel("Date" if 'date' in result_df else "Index")
        plt.ylabel("Value")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.legend()
        plt.show()