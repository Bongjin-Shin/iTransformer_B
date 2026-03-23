"""Discount Factor 적용 및 CUSUM 기반 Anomaly Detection (Validation 파일 활용 버전)"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.ndimage import uniform_filter1d

# ============== Configuration ==============
# 1. 실험 설정 (실행했던 .sh 파일의 설정과 맞춰주세요)
MODEL_NAME = "iTransformer"
DATA_ID = "D-14"        # 분석할 파일 ID (예: A-1, B-1 등)
SEQ_LEN = 96           # Lookback window size
PRED_LEN = 96          # Prediction length

# 2. CUSUM 및 탐지 설정
CHANNEL = 0            # Univariate이므로 0 고정
DISCOUNT_FACTOR = 0.9  # 최근 예측에 가중치 (0.9 권장)

# Threshold settings (For Gradient/Value warnings)
GRADIENT_THRESHOLD = 0.5    
VALUE_THRESHOLD = 1.5       
SMOOTHING_WINDOW = 5        
# ==================================================

def find_result_path(model_name, data_id, seq_len, pred_len):
    """
    results 폴더에서 조건에 맞는 가장 최신 실험 폴더를 찾습니다.
    """
    results_dir = './results/'
    if not os.path.exists(results_dir):
        print(f"Error: results folder not found: {results_dir}")
        return None
    
    # .sh에서 설정한 model_id 패턴에 맞춰 검색
    search_pattern = f"*{data_id}_seq{seq_len}_pred{pred_len}*"
    
    candidates = []
    for folder in os.listdir(results_dir):
        if data_id in folder and f"seq{seq_len}" in folder and f"pred{pred_len}" in folder:
            candidates.append(folder)
    
    if len(candidates) == 0:
        print(f"Error: Cannot find result folder for {data_id} with seq{seq_len}/pred{pred_len}")
        return None
    
    # 여러 개일 경우 가장 최근 것 선택
    candidates.sort()
    target_folder = candidates[-1]
    return os.path.join(results_dir, target_folder)

def load_data(base_path):
    """Load prediction results (Test & Validation)"""
    try:
        # Test Data
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        
        # Validation Data (존재 여부 확인)
        val_true_path = os.path.join(base_path, 'val_true.npy')
        val_pred_path = os.path.join(base_path, 'val_pred.npy')
        
        val_trues, val_preds = None, None
        if os.path.exists(val_true_path) and os.path.exists(val_pred_path):
            val_trues = np.load(val_true_path)
            val_preds = np.load(val_pred_path)
            print(f"Loaded VALIDATION data: {val_trues.shape}")
        else:
            print("Warning: Validation files (val_true/val_pred) not found!")

        print(f"Loaded TEST data: {trues.shape}")
        return trues, preds, val_trues, val_preds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def compute_reconstructed_error(trues, preds, channel, seq_len, pred_len, discount_factor):
    """
    Sliding Window 예측 결과를 시간 축으로 복원 (Weighted Average)
    """
    n_samples, p_len, n_feats = preds.shape
    total_time_steps = n_samples + seq_len + pred_len 
    
    mse_sum = np.zeros(total_time_steps)
    signed_sum = np.zeros(total_time_steps) 
    weight_sum = np.zeros(total_time_steps)
    pred_sum = np.zeros(total_time_steps)
    actual_values = np.full(total_time_steps, np.nan)
    
    for i in range(n_samples):
        start_t = i + seq_len 
        
        for j in range(p_len):
            t = start_t + j
            if t >= total_time_steps: break
            
            pred_val = preds[i, j, channel]
            true_val = trues[i, j, channel]
            weight = discount_factor ** j
            error = pred_val - true_val
            
            mse_sum[t] += weight * (error ** 2)
            signed_sum[t] += weight * error
            pred_sum[t] += weight * pred_val
            weight_sum[t] += weight
            
            if np.isnan(actual_values[t]):
                actual_values[t] = true_val

    valid_mask = weight_sum > 0
    max_idx = np.where(valid_mask)[0][-1]
    
    final_len = max_idx + 1
    avg_mse = np.full(final_len, np.nan)
    avg_signed = np.full(final_len, np.nan)
    avg_pred = np.full(final_len, np.nan)
    actual = np.full(final_len, np.nan)
    
    valid_slice = valid_mask[:final_len]
    avg_mse[valid_slice] = mse_sum[:final_len][valid_slice] / weight_sum[:final_len][valid_slice]
    avg_signed[valid_slice] = signed_sum[:final_len][valid_slice] / weight_sum[:final_len][valid_slice]
    avg_pred[valid_slice] = pred_sum[:final_len][valid_slice] / weight_sum[:final_len][valid_slice]
    actual[valid_slice] = actual_values[:final_len][valid_slice]
    
    return {
        'time_steps': np.arange(final_len),
        'actual': actual,
        'pred': avg_pred,
        'mse': avg_mse,
        'signed_error': avg_signed
    }

def cusum(error, target, b, mode='up'):
    S = np.zeros_like(error)
    for t in range(1, len(error)):
        if np.isnan(error[t]):
            S[t] = S[t-1]
            continue
        if mode == 'up':
            S[t] = max(0, S[t-1] + (error[t] - target - b))
        else:
            S[t] = min(0, S[t-1] + (error[t] - target + b))
    return S

def plot_analysis(res_test, cusum_res, channel, data_id):
    """Visualization"""
    
    time_steps = res_test['time_steps']
    actual = res_test['actual']
    pred = res_test['pred']
    # mse와 signed_error는 그래프에서 제외되므로 변수 할당은 필요하지만 plotting에는 사용 안 함
    
    # === 수정된 부분: subplots 3개 -> 2개로 변경 ===
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 1. Actual vs Prediction (첫 번째 그래프)
    axes[0].plot(time_steps, actual, 'k-', label='Actual', alpha=0.6)
    axes[0].plot(time_steps, pred, 'r--', label='Predicted', alpha=0.8)
    # 타이틀 수정 반영
    axes[0].set_title(f'[{data_id}] Weighted Mean Prediction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # (기존 2번째 그래프 Error Analysis 삭제됨)
    
    # 2. CUSUM Analysis (두 번째 그래프로 이동)
    c_up = cusum_res['up']
    c_down = cusum_res['down']
    ucl = cusum_res['ucl']
    lcl = cusum_res['lcl']
    
    # 인덱스 변경: axes[2] -> axes[1]
    axes[1].plot(time_steps, c_up, 'r-', label='CUSUM Up')
    axes[1].plot(time_steps, c_down, 'b-', label='CUSUM Down')
    axes[1].axhline(y=ucl, color='r', linestyle='--', label=f'UCL: {ucl:.2f}')
    axes[1].axhline(y=lcl, color='b', linestyle='--', label=f'LCL: {lcl:.2f}')
    
    # Highlight anomalies
    anom_up = np.where(c_up > ucl)[0]
    anom_down = np.where(c_down < lcl)[0]
    if len(anom_up) > 0:
        axes[1].scatter(time_steps[anom_up], c_up[anom_up], color='red', s=15, zorder=5)
    if len(anom_down) > 0:
        axes[1].scatter(time_steps[anom_down], c_down[anom_down], color='blue', s=15, zorder=5)
        
    axes[1].set_title(f'CUSUM Analysis')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'vis_{data_id}_seq{SEQ_LEN}_pred{PRED_LEN}.png')
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved: {save_path}")
    plt.show()

def main():
    print("=" * 60)
    print(f"NPY Dataset Anomaly Detection (ID: {DATA_ID})")
    print("=" * 60)
    
    # 1. 경로 찾기 및 데이터 로드
    result_path = find_result_path(MODEL_NAME, DATA_ID, SEQ_LEN, PRED_LEN)
    if result_path is None: return
    
    trues, preds, val_trues, val_preds = load_data(result_path)
    if trues is None: return
    
    # 2. Validation 데이터 분석 (Calibration)
    target_mean = 0
    sigma = 1
    ucl, lcl = 1, -1
    slack_b = 0
    
    if val_trues is not None and val_preds is not None:
        print("\n[Phase 1] Calibrating CUSUM parameters using Validation Set...")
        res_val = compute_reconstructed_error(val_trues, val_preds, CHANNEL, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
        
        valid_val_err = res_val['signed_error'][~np.isnan(res_val['signed_error'])]
        
        target_mean = np.mean(valid_val_err)
        sigma = np.std(valid_val_err)
        slack_b = sigma   # * 0.5  # Slack parameter
        
        # Validation Set에 대해 CUSUM을 돌려 최대값을 한계선으로 설정
        temp_up = cusum(valid_val_err, target_mean, slack_b, 'up')
        temp_down = cusum(valid_val_err, target_mean, slack_b, 'down')
        
        # 너무 타이트하지 않게 약간의 마진(예: + 10%) 혹은 Sigma Control Limit 사용
        # 여기서는 Validation Max 값에 3 Sigma를 더해 False Alarm 방지
        ucl = np.max(temp_up) + (3 * sigma)
        lcl = np.min(temp_down) - (3 * sigma)
        
        print(f"  - Target Mean: {target_mean:.4f}")
        print(f"  - Std (Sigma): {sigma:.4f}")
        print(f"  - Calculated UCL: {ucl:.4f}, LCL: {lcl:.4f}")
    else:
        print("\n[Warning] Validation data missing. Cannot calibrate accurately.")
        return

    # 3. Test 데이터 분석 (Detection)
    print("\n[Phase 2] Detecting Anomalies in Test Set...")
    res_test = compute_reconstructed_error(trues, preds, CHANNEL, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
    
    # 4. CUSUM 계산
    final_up = cusum(res_test['signed_error'], target_mean, slack_b, 'up')
    final_down = cusum(res_test['signed_error'], target_mean, slack_b, 'down')
    
    cusum_results = {
        'up': final_up,
        'down': final_down,
        'ucl': ucl,
        'lcl': lcl
    }
    
    # 5. 시각화
    plot_analysis(res_test, cusum_results, CHANNEL, DATA_ID)

if __name__ == "__main__":
    main()