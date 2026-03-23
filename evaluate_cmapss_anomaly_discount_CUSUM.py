"""Discount Factor 적용 및 CUSUM 기반 Anomaly Detection"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import uniform_filter1d

# ============== Configuration ==============
SEQ_LEN = 48           # Lookback window size
PRED_LEN = 48          # Prediction length
CHANNEL = 13           # Sensor index to analyze (0-13)
ENGINE_INDEX = 1       # Test engine index (0-19 → engines 81-100)

# Threshold settings
GRADIENT_THRESHOLD = 1.0    # Gradient threshold for warning
VALUE_THRESHOLD = 2.0       # Absolute MSE/MAE threshold
SMOOTHING_WINDOW = 5        # Window size for smoothing

# Discount factor for weighting predictions
DISCOUNT_FACTOR = 0.80       # e.g., 0.9 means recent

# Sensor names (14 sensors after filtering)
SENSOR_NAMES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
# ==================================================

def get_engine_info_all():
    """Load all engine information (61~100) for validation/test split"""
    try:
        df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
        all_engine_ids = list(range(61, 101))
        engine_info = {}
        for eid in all_engine_ids:
            engine_data = df[df['id'] == eid]
            engine_info[eid] = {
                'length': len(engine_data),
                'start_idx': 0
            }
        return engine_info, all_engine_ids
    except Exception as e:
        print(f"Failed to load engine info: {e}")
        return None, None

def get_sample_range_for_engine_id(target_eid, engine_info, all_engine_ids, seq_len, pred_len):
    sample_start = 0
    for eid in all_engine_ids:
        if eid == target_eid:
            break
        engine_len = engine_info[eid]['length']
        num_samples = engine_len - seq_len - pred_len + 1
        sample_start += max(0, num_samples)
    current_len = engine_info[target_eid]['length']
    num_samples = current_len - seq_len - pred_len + 1
    sample_end = sample_start + num_samples
    return sample_start, sample_end, num_samples, current_len

def find_result_path(seq_len, pred_len):
    """
    Find result folder matching seq_len and pred_len.
    Folder naming convention:
    - 'seq{SEQ}' contains seq_len
    - 'll{PRED}' contains pred_len
    """
    results_dir = './results/'
    
    if not os.path.exists(results_dir):
        print(f"Error: results folder not found: {results_dir}")
        return None
    
    seq_pattern = f'seq{seq_len}'
    pred_pattern = f'll{pred_len}'
    
    candidates = []
    for folder in os.listdir(results_dir):
        if 'CMAPSS' in folder and 'Progressive' not in folder and 'Expanding' not in folder:
            if seq_pattern in folder and pred_pattern in folder:
                candidates.append(folder)
    
    if len(candidates) == 1:
        return os.path.join(results_dir, candidates[0])
    elif len(candidates) > 1:
        print(f"Warning: Multiple folders found for seq_len={seq_len}, pred_len={pred_len}")
        for c in candidates:
            print(f"  - {c}")
        return os.path.join(results_dir, sorted(candidates)[-1])
    
    print(f"Error: Cannot find result folder with seq_len={seq_len}, pred_len={pred_len}")
    return None


def load_data(base_path):
    """Load prediction results"""
    try:
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        print(f"Loaded TEST data shape - trues: {trues.shape}, preds: {preds.shape}")
        val_trues = np.load(os.path.join(base_path, 'val_true.npy'))
        val_preds = np.load(os.path.join(base_path, 'val_pred.npy'))
        print(f"Loaded VALIDATION data shape - trues: {val_trues.shape}, preds: {val_preds.shape}")
        return trues, preds, val_trues, val_preds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


def get_engine_info():
    """Load test engine information from data"""
    try:
        df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
        test_engine_ids = list(range(81, 101))  # Engine 81-100 for test
        
        engine_info = {}
        for eid in test_engine_ids:
            engine_data = df[df['id'] == eid]
            engine_info[eid] = {
                'length': len(engine_data),
                'start_idx': 0
            }
        return engine_info, test_engine_ids
    except Exception as e:
        print(f"Failed to load engine info: {e}")
        return None, None


def get_sample_range_for_engine(engine_idx, engine_info, test_engine_ids, seq_len, pred_len):
    """Calculate sample range for a specific engine"""
    # Calculate samples per engine
    sample_start = 0
    for i in range(engine_idx):
        eid = test_engine_ids[i]
        engine_len = engine_info[eid]['length']
        num_samples = engine_len - seq_len - pred_len + 1
        sample_start += max(0, num_samples)
    
    current_eid = test_engine_ids[engine_idx]
    current_len = engine_info[current_eid]['length']
    num_samples = current_len - seq_len - pred_len + 1
    sample_end = sample_start + num_samples
    
    return sample_start, sample_end, current_eid, current_len


def compute_overlapping_predictions_discount(trues, preds, sample_start, sample_end, channel, seq_len, pred_len, engine_len, discount_factor):
    engine_trues = trues[sample_start:sample_end]
    engine_preds = preds[sample_start:sample_end]
    n_samples = engine_trues.shape[0]
    total_time_steps = engine_len
    mse_sum = np.zeros(total_time_steps)
    mae_sum = np.zeros(total_time_steps)
    signed_sum = np.zeros(total_time_steps)
    weight_sum = np.zeros(total_time_steps)
    pred_count = np.zeros(total_time_steps, dtype=int)  # 각 시점별 겹치는 예측 개수
    pred_sum = np.zeros(total_time_steps)  # discount factor를 곱한 예측값의 합
    actual_values = np.full(total_time_steps, np.nan)
    for sample_idx in range(n_samples):
        start_t = seq_len + sample_idx
        for j in range(pred_len):
            t = start_t + j
            if t < total_time_steps:
                pred_val = engine_preds[sample_idx, j, channel]
                true_val = engine_trues[sample_idx, j, channel]
                distance = j  # j=0: 최근, j=pred_len-1: 가장 먼 미래
                weight = discount_factor ** distance
                error = pred_val - true_val
                mse_sum[t] += weight * (error ** 2)
                mae_sum[t] += weight * np.abs(error)
                signed_sum[t] += weight * error
                weight_sum[t] += weight
                pred_sum[t] += weight * pred_val
                pred_count[t] += 1
                if np.isnan(actual_values[t]):
                    actual_values[t] = true_val
    valid_mask = weight_sum > 0
    avg_mse = np.full(total_time_steps, np.nan)
    avg_mae = np.full(total_time_steps, np.nan)
    avg_signed = np.full(total_time_steps, np.nan)
    avg_pred = np.full(total_time_steps, np.nan)
    avg_mse[valid_mask] = mse_sum[valid_mask] / weight_sum[valid_mask]
    avg_mae[valid_mask] = mae_sum[valid_mask] / weight_sum[valid_mask]
    avg_signed[valid_mask] = signed_sum[valid_mask] / weight_sum[valid_mask]
    avg_pred[valid_mask] = pred_sum[valid_mask] / weight_sum[valid_mask]
    full_overlap_start = seq_len + pred_len - 1
    return {
        'time_steps': np.arange(total_time_steps),
        'actual': actual_values,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'avg_signed': avg_signed,
        'avg_pred': avg_pred,
        'weight_sum': weight_sum,
        'pred_count': pred_count,
        'full_overlap_start': full_overlap_start,
        'n_samples': n_samples
    }

def compute_pointwise_error(result, seq_len):
    """
    Extract point-wise MSE and MAE from the result.
    Method 2: These are already averaged errors from individual predictions.
    
    Now includes all time steps from seq_len (where predictions start)
    to dynamically use however many predictions are available at each time step.
    """
    avg_mse = result['avg_mse']
    avg_mae = result['avg_mae']
    avg_signed = result['avg_signed']
    pred_count = result['pred_count']
    n = len(avg_mse)
    
    # Use values from seq_len (where predictions start)
    # This includes both ramp-up (count 1→pred_len) and ramp-down regions
    mse = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    signed_error = np.full(n, np.nan)
    
    for t in range(seq_len, n):
        if pred_count[t] > 0:  # Only where predictions exist
            mse[t] = avg_mse[t]
            mae[t] = avg_mae[t]
            signed_error[t] = avg_signed[t]
    
    return mse, mae, signed_error


def compute_gradient(values, window=5):
    """
    Compute gradient (rate of change) of values.
    Uses smoothing to reduce noise sensitivity.
    """
    # Smooth the values first
    valid_mask = ~np.isnan(values)
    smoothed = np.full_like(values, np.nan)
    
    if np.sum(valid_mask) > window:
        valid_values = values[valid_mask]
        smoothed_valid = uniform_filter1d(valid_values, size=window)
        smoothed[valid_mask] = smoothed_valid
    
    # Compute gradient
    gradient = np.full_like(values, np.nan)
    valid_idx = np.where(valid_mask)[0]
    
    for i in range(1, len(valid_idx)):
        t_curr = valid_idx[i]
        t_prev = valid_idx[i-1]
        if t_curr - t_prev == 1:  # Consecutive
            gradient[t_curr] = smoothed[t_curr] - smoothed[t_prev]
    
    return gradient, smoothed


def detect_anomaly(smoothed_mse, smoothed_mae, gradient_mse, gradient_mae, 
                   gradient_threshold, value_threshold, full_overlap_start):
    """
    Detect anomaly based on gradient and value thresholds.
    Uses smoothed values for consistent detection.
    Returns warning time steps and warning messages.
    """
    warnings = []
    
    for t in range(full_overlap_start, len(smoothed_mse)):
        warning_msg = []
        
        # Check gradient threshold
        if not np.isnan(gradient_mse[t]) and gradient_mse[t] > gradient_threshold:
            warning_msg.append(f"MSE gradient ({gradient_mse[t]:.4f}) > threshold ({gradient_threshold})")
        
        if not np.isnan(gradient_mae[t]) and gradient_mae[t] > gradient_threshold:
            warning_msg.append(f"MAE gradient ({gradient_mae[t]:.4f}) > threshold ({gradient_threshold})")
        
        # Check absolute value threshold (using smoothed values)
        if not np.isnan(smoothed_mse[t]) and smoothed_mse[t] > value_threshold:
            warning_msg.append(f"Smoothed MSE ({smoothed_mse[t]:.4f}) > threshold ({value_threshold})")
        
        if not np.isnan(smoothed_mae[t]) and smoothed_mae[t] > value_threshold:
            warning_msg.append(f"Smoothed MAE ({smoothed_mae[t]:.4f}) > threshold ({value_threshold})")
        
        if warning_msg:
            warnings.append({
                'time_step': t,
                'messages': warning_msg
            })
    
    return warnings


def plot_analysis(result, mse, mae, signed_error, gradient_mse, gradient_mae, smoothed_mse, smoothed_mae,
                  warnings, channel, engine_idx, seq_len, pred_len,
                  gradient_threshold, value_threshold):
    """
    Plot comprehensive analysis:
    1. Actual vs Averaged Prediction
    2. Point-wise MSE/MAE
    3. Gradient of MSE/MAE with thresholds
    4. CUSUM
    """
    # CUSUM만 단일 플롯으로 시각화
    time_steps = result['time_steps']
    plt.figure(figsize=(16, 9))
    if hasattr(plot_analysis, 'cusum_up') and hasattr(plot_analysis, 'cusum_down'):
        cusum_up = plot_analysis.cusum_up
        cusum_down = plot_analysis.cusum_down
        UCL = plot_analysis.UCL
        LCL = plot_analysis.LCL
        plt.plot(time_steps, cusum_up, 'r-', label='CUSUM Up', linewidth=1.5)
        plt.plot(time_steps, cusum_down, 'b-', label='CUSUM Down', linewidth=1.5)
        plt.axhline(y=UCL, color='r', linestyle='--', label=f'UCL: {UCL:.4f}')
        plt.axhline(y=LCL, color='b', linestyle='--', label=f'LCL: {LCL:.4f}')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        # UCL/LCL 넘는 warning points 마크
        cusum_up_warnings = [t for t in range(len(cusum_up)) if cusum_up[t] > UCL]
        cusum_down_warnings = [t for t in range(len(cusum_down)) if cusum_down[t] < LCL]
        if cusum_up_warnings:
            plt.scatter([time_steps[t] for t in cusum_up_warnings], [cusum_up[t] for t in cusum_up_warnings],
                        color='red', s=20, zorder=5, label='CUSUM_UP Warning')
        if cusum_down_warnings:
            plt.scatter([time_steps[t] for t in cusum_down_warnings], [cusum_down[t] for t in cusum_down_warnings],
                        color='blue', s=20, zorder=5, label='CUSUM_DOWN Warning')
        plt.title(f'CUSUM_Discount Factor : {DISCOUNT_FACTOR}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No CUSUM Information', ha='center', va='center', fontsize=14)
        plt.title('CUSUM')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Save figure
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'anomaly_detection_engine{engine_idx+81}_ch{channel}_DF{DISCOUNT_FACTOR}_CUSUM.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("C-MAPSS Anomaly Detection based on Prediction Error (Corrected Indexing)")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  SEQ_LEN = {SEQ_LEN}")
    print(f"  PRED_LEN = {PRED_LEN}")
    print(f"  CHANNEL = {CHANNEL} ({SENSOR_NAMES[CHANNEL] if CHANNEL < len(SENSOR_NAMES) else 'Unknown'})")
    print(f"  ENGINE_INDEX = {ENGINE_INDEX} (Engine {ENGINE_INDEX + 81})")
    print(f"  GRADIENT_THRESHOLD = {GRADIENT_THRESHOLD}")
    print(f"  VALUE_THRESHOLD = {VALUE_THRESHOLD}")
    print(f"  SMOOTHING_WINDOW = {SMOOTHING_WINDOW}")
    print()
    
    # Find and load data
    base_path = find_result_path(SEQ_LEN, PRED_LEN)
    if base_path is None:
        return
    print(f"Using result path: {base_path}")
    
    # Load separate files for Test and Validation
    trues, preds, val_trues, val_preds = load_data(base_path)
    if trues is None or val_trues is None:
        print("Error: Failed to load data files.")
        return
    
    # Load all engine info (validation+test)
    engine_info, all_engine_ids = get_engine_info_all()
    if engine_info is None:
        return
        
    # Define ID lists
    val_ids = list(range(61, 81))   # Engine 61~80 (Validation File)
    test_ids = list(range(81, 101)) # Engine 81~100 (Test File)
    
    print(f"\nValidation engines: {val_ids}")
    print(f"Test engines: {test_ids}")

    # =========================================================================
    # 1. Validation set에서 MAE 기반 CUSUM UCL/LCL 계산 (val_ids 기준)
    # =========================================================================
    val_mae_list = []
    print("\n[Phase 1] Processing Validation Data...")
    
    for vid in val_ids:
        # [중요] val_trues 파일은 Engine 61이 0번부터 시작하므로 'val_ids'를 넘겨야 함
        s_start, s_end, n_samples, e_len = get_sample_range_for_engine_id(
            vid, engine_info, val_ids, SEQ_LEN, PRED_LEN
        )
        
        if s_end > val_trues.shape[0]:
            print(f"⚠️  Warning: Validation Data for Engine {vid} is missing/truncated! (File len: {val_trues.shape[0]}, Req: {s_end})")
            continue
            
        if n_samples > 0:
            result_val = compute_overlapping_predictions_discount(
                val_trues, val_preds, s_start, s_end, CHANNEL, SEQ_LEN, PRED_LEN, e_len, DISCOUNT_FACTOR
            )
            # Method 2: signed error (부호 있는 error) 사용
            _, _, val_signed = compute_pointwise_error(result_val, SEQ_LEN)
            val_mae_list.append(val_signed[~np.isnan(val_signed)])
            
    if not val_mae_list:
        print("Error: No validation data extracted. Cannot compute CUSUM thresholds.")
        return

    # CUSUM Parameters Calculation
    all_val_mae = np.concatenate(val_mae_list)
    Target = np.mean(all_val_mae)
    sigma_val = np.std(all_val_mae)
    b = sigma_val * 0.05  # Slack parameter
    
    # Define CUSUM function locally to capture parameters if needed, or use global
    def cusum_custom_local(error, Target, b, mode='up'):
        S = np.zeros_like(error)
        for t in range(1, len(error)):
            if np.isnan(error[t]):
                S[t] = S[t-1]
                continue
            if mode == 'up':
                S[t] = max(0, S[t-1] + (error[t] - Target - b))
            else:
                S[t] = min(0, S[t-1] + (error[t] - Target + b))
        return S

    val_max_cusum_up = -np.inf
    val_min_cusum_down = np.inf
    
    for mae_segment in val_mae_list:
        c_up = cusum_custom_local(mae_segment, Target, b, mode='up')
        c_down = cusum_custom_local(mae_segment, Target, b, mode='down')
        val_max_cusum_up = max(val_max_cusum_up, np.max(c_up))
        val_min_cusum_down = min(val_min_cusum_down, np.min(c_down))
        
    UCL = val_max_cusum_up
    LCL = val_min_cusum_down
    
    print(f"\n[CUSUM Parameters from Validation]")
    print(f"  - Target (mean MAE): {Target:.6f}")
    print(f"  - Std (sigma): {sigma_val:.6f}")
    print(f"  - b (sigma*0.05): {b:.6f}")
    print(f"  - UCL (Val Max): {UCL:.6f}")
    print(f"  - LCL (Val Min): {LCL:.6f}")

    # =========================================================================
    # 2. Test set 분석 (test_ids 기준)
    # =========================================================================
    if ENGINE_INDEX >= len(test_ids):
        print(f"Error: ENGINE_INDEX {ENGINE_INDEX} out of range (max: {len(test_ids)-1})")
        return
        
    target_eid = test_ids[ENGINE_INDEX]
    
    # [중요] trues 파일은 Engine 81이 0번부터 시작하므로 'test_ids'를 넘겨야 함
    s_start, s_end, n_samples, engine_len = get_sample_range_for_engine_id(
        target_eid, engine_info, test_ids, SEQ_LEN, PRED_LEN
    )
    
    print(f"\nAnalyzing Test Engine {target_eid}")
    print(f"  Engine length: {engine_len} cycles")
    print(f"  Sample range: {s_start} to {s_end-1} ({s_end - s_start} samples)")
    
    # 파일 범위 체크
    if s_start >= trues.shape[0]:
        print(f"Error: Start index ({s_start}) exceeds file size ({trues.shape[0]}). Check indexing logic.")
        return
    if s_end > trues.shape[0]:
        print(f"Warning: End index ({s_end}) exceeds file size. Clipping to {trues.shape[0]}.")
        s_end = trues.shape[0]

    result = compute_overlapping_predictions_discount(
        trues, preds, s_start, s_end, CHANNEL, SEQ_LEN, PRED_LEN, engine_len, DISCOUNT_FACTOR
    )
    mse, mae, signed_error = compute_pointwise_error(result, SEQ_LEN)
    
    # Compute gradients
    gradient_mse, smoothed_mse = compute_gradient(mse, SMOOTHING_WINDOW)
    gradient_mae, smoothed_mae = compute_gradient(mae, SMOOTHING_WINDOW)

    # 기존 방식 anomaly 탐지
    warnings = detect_anomaly(smoothed_mse, smoothed_mae, gradient_mse, gradient_mae,
                              GRADIENT_THRESHOLD, VALUE_THRESHOLD, result['full_overlap_start'])

    # CUSUM 기반 anomaly 탐지
    valid_signed = signed_error[~np.isnan(signed_error)]
    cusum_up = None
    cusum_down = None
    
    # Method 2: signed error(부호 있는 error)로 CUSUM 계산
    if len(valid_signed) > 0:
        cusum_up = cusum_custom_local(signed_error, Target, b, mode='up')
        cusum_down = cusum_custom_local(signed_error, Target, b, mode='down')
        
        cusum_warnings = []
        for t in range(len(signed_error)):
            if cusum_up[t] > UCL:
                cusum_warnings.append({'time_step': t, 'msg': 'CUSUM_UP'})
            elif cusum_down[t] < LCL:
                cusum_warnings.append({'time_step': t, 'msg': 'CUSUM_DOWN'})
        print(f"\n{'=' * 60}")
        print(f"CUSUM ANOMALY DETECTION RESULTS")
        print(f"{'=' * 60}")
        if cusum_warnings:
            print(f"\n⚠️  {len(cusum_warnings)} CUSUM WARNING(S) DETECTED!")
            first_cusum = cusum_warnings[0]
            last_cusum = cusum_warnings[-1]
            print(f"  Range: t={first_cusum['time_step']} to t={last_cusum['time_step']}")
        else:
            print("\n✅ No CUSUM anomalies detected within the thresholds.")
    else:
        print("\n❌ Error: No valid signed error calculated for Test Engine. Cannot run CUSUM.")
        return

    # 시각화 함수에 데이터 전달
    plot_analysis.cusum_up = cusum_up
    plot_analysis.cusum_down = cusum_down
    plot_analysis.UCL = UCL
    plot_analysis.LCL = LCL
    
    # Plot analysis
    plot_analysis(result, mse, mae, signed_error, gradient_mse, gradient_mae, smoothed_mse, smoothed_mae,
                  warnings, CHANNEL, ENGINE_INDEX, SEQ_LEN, PRED_LEN,
                  GRADIENT_THRESHOLD, VALUE_THRESHOLD)


if __name__ == "__main__":
    main()