"""Threshold 거는 시각화 코드 (Discount Factor 적용 버전)"""

#!/usr/bin/env python3
"""
C-MAPSS Anomaly Detection based on Prediction Error

Logic:
1. At each time step t, we have pred_len predictions pointing to it
   (from t-pred_len to t-1 predictions)
2. Average these pred_len predictions and compare with actual value
3. Calculate point-wise MSE/MAE at each time step
4. If the gradient of MSE/MAE exceeds threshold → Warning!

This approach detects when the engine starts deviating from normal patterns.
"""

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
DISCOUNT_FACTOR = 0.95       # e.g., 0.9 means recent

# Sensor names (14 sensors after filtering)
SENSOR_NAMES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
# ==================================================


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
        print(f"Loaded data shape - trues: {trues.shape}, preds: {preds.shape}")
        return trues, preds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None


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
                weight_sum[t] += weight
                pred_sum[t] += weight * pred_val
                pred_count[t] += 1
                if np.isnan(actual_values[t]):
                    actual_values[t] = true_val
    valid_mask = weight_sum > 0
    avg_mse = np.full(total_time_steps, np.nan)
    avg_mae = np.full(total_time_steps, np.nan)
    avg_pred = np.full(total_time_steps, np.nan)
    avg_mse[valid_mask] = mse_sum[valid_mask] / weight_sum[valid_mask]
    avg_mae[valid_mask] = mae_sum[valid_mask] / weight_sum[valid_mask]
    avg_pred[valid_mask] = pred_sum[valid_mask] / weight_sum[valid_mask]
    full_overlap_start = seq_len + pred_len - 1
    return {
        'time_steps': np.arange(total_time_steps),
        'actual': actual_values,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
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
    pred_count = result['pred_count']
    n = len(avg_mse)
    
    # Use values from seq_len (where predictions start)
    # This includes both ramp-up (count 1→pred_len) and ramp-down regions
    mse = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    
    for t in range(seq_len, n):
        if pred_count[t] > 0:  # Only where predictions exist
            mse[t] = avg_mse[t]
            mae[t] = avg_mae[t]
    
    return mse, mae


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


def plot_analysis(result, mse, mae, gradient_mse, gradient_mae, smoothed_mse, smoothed_mae,
                  warnings, channel, engine_idx, seq_len, pred_len,
                  gradient_threshold, value_threshold):
    """
    Plot comprehensive analysis:
    1. Actual vs Averaged Prediction
    2. Point-wise MSE/MAE
    3. Gradient of MSE/MAE with thresholds
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
    time_steps = result['time_steps']
    actual = result['actual']
    avg_pred = result['avg_pred']
    pred_count = result['pred_count']
    full_overlap_start = result['full_overlap_start']
    n_samples = result['n_samples']
    
    # Full overlap ends at seq_len + n_samples - 1 (last sample's first prediction)
    full_overlap_end = seq_len + n_samples - 1
    
    sensor_name = SENSOR_NAMES[channel] if channel < len(SENSOR_NAMES) else f"Ch{channel}"
    
    # ===== Plot 1: Actual vs Averaged Prediction =====
    ax1 = axes[0]
    ax1.plot(time_steps, actual, 'b-', label='Actual', linewidth=1.5, alpha=0.8)
    ax1.plot(time_steps, avg_pred, 'r--', label='Avg Prediction', linewidth=1.5, alpha=0.8)
    ax1.axvline(x=full_overlap_start, color='green', linestyle=':', linewidth=2, 
                label=f'Full overlap start (t={full_overlap_start})')
    ax1.axvline(x=full_overlap_end, color='green', linestyle=':', linewidth=2,
                label=f'Full overlap end (t={full_overlap_end})')
    ax1.axvline(x=seq_len, color='purple', linestyle=':', linewidth=1.5,
                label=f'seq_len={seq_len}')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel(f'{sensor_name} Value')
    ax1.set_title(f'Engine {engine_idx+81} - Sensor {sensor_name}: Actual vs Averaged Prediction')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # ===== Plot 2: Prediction Count (how many predictions overlap) =====
    ax2 = axes[1]
    ax2.fill_between(time_steps, pred_count, alpha=0.5, color='orange')
    ax2.axhline(y=pred_len, color='red', linestyle='--', label=f'Max overlap = pred_len ({pred_len})')
    ax2.axvline(x=full_overlap_start, color='green', linestyle=':', linewidth=2)
    ax2.axvline(x=full_overlap_end, color='green', linestyle=':', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Prediction Count')
    ax2.set_title('Number of Overlapping Predictions at Each Time Step')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # ===== Plot 3: Point-wise MSE/MAE =====
    ax3 = axes[2]
    ax3.plot(time_steps, mse, 'r-', label='Avg MSE', linewidth=1, alpha=0.5)
    ax3.plot(time_steps, mae, 'b-', label='Avg MAE', linewidth=1, alpha=0.5)
    ax3.plot(time_steps, smoothed_mse, 'r-', label='Smoothed Avg MSE', linewidth=2)
    ax3.plot(time_steps, smoothed_mae, 'b-', label='Smoothed Avg MAE', linewidth=2)
    ax3.axhline(y=value_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Value Threshold ({value_threshold})')
    ax3.axvline(x=full_overlap_start, color='green', linestyle=':', linewidth=2)
    ax3.axvline(x=full_overlap_end, color='green', linestyle=':', linewidth=2)
    
    # Mark value threshold violations (based on smoothed MSE)
    value_warning_times = []
    for t in range(full_overlap_start, len(smoothed_mse)):
        if not np.isnan(smoothed_mse[t]) and smoothed_mse[t] > value_threshold:
            value_warning_times.append(t)
    
    if value_warning_times:
        ax3.scatter(value_warning_times, 
                   [smoothed_mse[t] for t in value_warning_times],
                   color='red', s=30, zorder=5, alpha=0.7, label='Value Warning Points')
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Error')
    ax3.set_title('Averaged MSE/MAE from Individual Predictions (Method 2)')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # ===== Plot 4: Gradient of MSE/MAE =====
    ax4 = axes[3]
    ax4.plot(time_steps, gradient_mse, 'r-', label='MSE Gradient', linewidth=1.5)
    ax4.plot(time_steps, gradient_mae, 'b-', label='MAE Gradient', linewidth=1.5)
    ax4.axhline(y=gradient_threshold, color='black', linestyle='--', linewidth=2,
                label=f'Gradient Threshold ({gradient_threshold})')
    ax4.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax4.axvline(x=full_overlap_start, color='green', linestyle=':', linewidth=2)
    ax4.axvline(x=full_overlap_end, color='green', linestyle=':', linewidth=2)
    
    # Mark warning points - only for gradient threshold violations
    gradient_warning_times = []
    for t in range(full_overlap_start, len(gradient_mse)):
        if (not np.isnan(gradient_mse[t]) and gradient_mse[t] > gradient_threshold) or \
           (not np.isnan(gradient_mae[t]) and gradient_mae[t] > gradient_threshold):
            gradient_warning_times.append(t)
    
    if gradient_warning_times:
        ax4.scatter(gradient_warning_times, 
                   [gradient_mse[t] if not np.isnan(gradient_mse[t]) else 0 for t in gradient_warning_times],
                   color='red', s=50, zorder=5, label='Gradient Warning Points')
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Gradient')
    ax4.set_title('Gradient of Averaged MSE/MAE')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_dir = './figures'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'anomaly_detection_engine{engine_idx+81}_ch{channel}_DF{DISCOUNT_FACTOR}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {save_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("C-MAPSS Anomaly Detection based on Prediction Error")
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
    
    trues, preds = load_data(base_path)
    if trues is None:
        return
    
    # Get engine information from CSV
    engine_info, test_engine_ids = get_engine_info()
    if engine_info is None:
        return
    
    print(f"\nTest engines: {test_engine_ids}")
    for eid in test_engine_ids:
        print(f"  Engine {eid}: {engine_info[eid]['length']} cycles")
    
    if ENGINE_INDEX >= len(test_engine_ids):
        print(f"Error: ENGINE_INDEX {ENGINE_INDEX} out of range (max: {len(test_engine_ids)-1})")
        return
    
    # Get sample range for this engine
    sample_start, sample_end, current_eid, engine_len = get_sample_range_for_engine(
        ENGINE_INDEX, engine_info, test_engine_ids, SEQ_LEN, PRED_LEN
    )
    
    print(f"\nAnalyzing Engine {current_eid}")
    print(f"  Engine length: {engine_len} cycles")
    print(f"  Sample range: {sample_start} to {sample_end-1} ({sample_end - sample_start} samples)")
    
    # Compute overlapping predictions
    result = compute_overlapping_predictions_discount(
        trues, preds, sample_start, sample_end, CHANNEL, SEQ_LEN, PRED_LEN, engine_len, DISCOUNT_FACTOR
    )
    print(f"  Total time steps: {len(result['time_steps'])}")
    print(f"  Full overlap starts at: t={result['full_overlap_start']}")
    
    # Compute point-wise errors (Method 2: averaged errors from individual predictions)
    # Now includes all time steps from SEQ_LEN (dynamic overlap count)
    mse, mae = compute_pointwise_error(result, SEQ_LEN)
    
    # Compute gradients
    gradient_mse, smoothed_mse = compute_gradient(mse, SMOOTHING_WINDOW)
    gradient_mae, smoothed_mae = compute_gradient(mae, SMOOTHING_WINDOW)
    
    # Detect anomalies (using smoothed values)
    warnings = detect_anomaly(smoothed_mse, smoothed_mae, gradient_mse, gradient_mae,
                              GRADIENT_THRESHOLD, VALUE_THRESHOLD, result['full_overlap_start'])
    
    # Print warnings
    print(f"\n{'=' * 60}")
    print(f"ANOMALY DETECTION RESULTS")
    print(f"{'=' * 60}")
    
    if warnings:
        print(f"\n⚠️  {len(warnings)} WARNING(S) DETECTED!")
        print("-" * 40)
        
        # Group consecutive warnings
        first_warning = warnings[0]
        last_warning = warnings[-1]
        
        print(f"\nFirst warning at time step: {first_warning['time_step']}")
        for msg in first_warning['messages']:
            print(f"  → {msg}")
        
        if len(warnings) > 1:
            print(f"\nLast warning at time step: {last_warning['time_step']}")
            for msg in last_warning['messages']:
                print(f"  → {msg}")
        
        print(f"\nTotal warning time steps: {len(warnings)}")
        print(f"Warning range: t={first_warning['time_step']} to t={last_warning['time_step']}")
        
        # Estimate remaining useful life
        engine_end = len(result['time_steps']) - 1
        first_warning_time = first_warning['time_step']
        remaining_after_warning = engine_end - first_warning_time
        print(f"\n📊 Engine ends at time step: {engine_end}")
        print(f"📊 Remaining time after first warning: {remaining_after_warning} time steps")
    else:
        print("\n✅ No anomalies detected within the thresholds.")
    
    # Plot analysis
    plot_analysis(result, mse, mae, gradient_mse, gradient_mae, smoothed_mse, smoothed_mae,
                  warnings, CHANNEL, ENGINE_INDEX, SEQ_LEN, PRED_LEN,
                  GRADIENT_THRESHOLD, VALUE_THRESHOLD)


if __name__ == "__main__":
    main()