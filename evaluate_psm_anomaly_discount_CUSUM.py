"""Discount Factor 적용 및 CUSUM 기반 Anomaly Detection (PSM Dataset)
각 센서 변수별로 독립적으로 UCL/LCL을 설정하고 CUSUM Thresholding 수행.
Vectorized implementation for performance.
"""

### 데이터 전체 plot에 대해서 CUSUM 적용한 그래프 뽑는 코드

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============== Configuration ==============
SEQ_LEN = 96           # Lookback window size
PRED_LEN = 96          # Prediction length

# Discount factor for weighting predictions
DISCOUNT_FACTOR = 0.9

# Sensor names (25 sensors, Label excluded)
SENSOR_NAMES = [f'feature_{i}' for i in range(25)]
NUM_CHANNELS = len(SENSOR_NAMES)

# PSM CSV path (for ground truth labels)
CSV_PATH = './dataset/TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv'

# Data split boundaries
TRAIN_END = 50001
VAL_END = 60001
# ==================================================


def find_result_path(seq_len, pred_len):
    """Find result folder matching PSM."""
    results_dir = './results/'
    if not os.path.exists(results_dir):
        print(f"Error: results folder not found: {results_dir}")
        return None
    candidates = [f for f in os.listdir(results_dir)
                  if 'PSM' in f and f'seq{seq_len}' in f and f'pred{pred_len}' in f]
    if len(candidates) == 0:
        print(f"Error: Cannot find PSM result folder with seq_len={seq_len}, pred_len={pred_len}")
        return None
    return os.path.join(results_dir, sorted(candidates)[-1])


def load_data(base_path):
    """Load prediction results (Test & Validation)"""
    try:
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        print(f"Loaded TEST data: trues {trues.shape}, preds {preds.shape}")
        val_trues = np.load(os.path.join(base_path, 'val_true.npy'))
        val_preds = np.load(os.path.join(base_path, 'val_pred.npy'))
        print(f"Loaded VAL  data: trues {val_trues.shape}, preds {val_preds.shape}")
        return trues, preds, val_trues, val_preds
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None


def compute_reconstructed_error_vectorized(trues, preds, channel, seq_len, pred_len, discount_factor):
    """
    Vectorized version: reconstruct time-aligned weighted signed error.
    Much faster than the nested Python loop.
    """
    n_samples = preds.shape[0]
    total_time_steps = n_samples + seq_len + pred_len

    # Discount weights: shape (pred_len,)
    weights = discount_factor ** np.arange(pred_len)

    # Errors for the selected channel: shape (n_samples, pred_len)
    errors = preds[:, :, channel] - trues[:, :, channel]
    pred_vals = preds[:, :, channel]
    true_vals = trues[:, :, channel]

    # Weighted errors/preds: shape (n_samples, pred_len)
    w_signed = errors * weights[np.newaxis, :]
    w_pred = pred_vals * weights[np.newaxis, :]
    w_weight = np.tile(weights, (n_samples, 1))  # (n_samples, pred_len)

    # Accumulate using np.add.at
    signed_sum = np.zeros(total_time_steps)
    pred_sum = np.zeros(total_time_steps)
    weight_sum = np.zeros(total_time_steps)
    actual_values = np.full(total_time_steps, np.nan)

    for j in range(pred_len):
        # Time indices: sample i → time (i + seq_len + j) for horizon j
        t_indices = np.arange(n_samples) + seq_len + j
        valid = t_indices < total_time_steps
        t_valid = t_indices[valid]
        np.add.at(signed_sum, t_valid, w_signed[valid, j])
        np.add.at(pred_sum, t_valid, w_pred[valid, j])
        np.add.at(weight_sum, t_valid, w_weight[valid, j])

    # Actual values: use the first occurrence for each time step
    for i in range(n_samples):
        t = i + seq_len
        if t < total_time_steps and np.isnan(actual_values[t]):
            actual_values[t] = true_vals[i, 0]

    valid_mask = weight_sum > 0
    max_idx = np.where(valid_mask)[0][-1] if np.any(valid_mask) else 0
    final_len = max_idx + 1

    avg_signed = np.full(final_len, np.nan)
    avg_pred = np.full(final_len, np.nan)
    actual = np.full(final_len, np.nan)

    vs = valid_mask[:final_len]
    avg_signed[vs] = signed_sum[:final_len][vs] / weight_sum[:final_len][vs]
    avg_pred[vs] = pred_sum[:final_len][vs] / weight_sum[:final_len][vs]
    actual[vs] = actual_values[:final_len][vs]

    return {
        'time_steps': np.arange(final_len),
        'actual': actual,
        'pred': avg_pred,
        'signed_error': avg_signed,
    }


def cusum(error, target, b, mode='up'):
    """CUSUM control chart"""
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


def load_ground_truth_labels():
    """Load ground truth anomaly labels from PSM CSV"""
    try:
        df = pd.read_csv(CSV_PATH)
        return df['Label'].values
    except Exception as e:
        print(f"Warning: Cannot load ground truth labels: {e}")
        return None


def plot_all_channels(all_results, gt_labels_test=None):
    """전체 채널 CUSUM 결과 한 장으로 시각화."""
    fig, axes = plt.subplots(NUM_CHANNELS, 1, figsize=(20, 3 * NUM_CHANNELS), sharex=True)
    fig.suptitle(f'PSM CUSUM per Sensor (Discount={DISCOUNT_FACTOR})',
                 fontsize=14, fontweight='bold', y=0.995)

    for ch in range(NUM_CHANNELS):
        ax = axes[ch]
        res = all_results[ch]
        if res is None:
            ax.text(0.5, 0.5, f'{SENSOR_NAMES[ch]}: SKIPPED', ha='center', va='center')
            continue

        ts = res['time_steps']
        c_up, c_down = res['cusum_up'], res['cusum_down']
        ucl, lcl = res['ucl'], res['lcl']

        ax.plot(ts, c_up, 'r-', linewidth=0.8, label='CUSUM Up')
        ax.plot(ts, c_down, 'b-', linewidth=0.8, label='CUSUM Down')
        ax.axhline(y=ucl, color='r', linestyle='--', linewidth=0.8, label=f'UCL: {ucl:.4f}')
        ax.axhline(y=lcl, color='b', linestyle='--', linewidth=0.8, label=f'LCL: {lcl:.4f}')
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.3)

        anom_up = np.where(c_up > ucl)[0]
        anom_down = np.where(c_down < lcl)[0]
        if len(anom_up) > 0:
            ax.scatter(ts[anom_up], c_up[anom_up], color='red', s=0.1, zorder=5)
        if len(anom_down) > 0:
            ax.scatter(ts[anom_down], c_down[anom_down], color='blue', s=0.1, zorder=5)

        if gt_labels_test is not None:
            gt_len = min(len(gt_labels_test), len(ts))
            ymin, ymax = ax.get_ylim()
            ax.fill_between(ts[:gt_len], ymin, ymax,
                            where=gt_labels_test[:gt_len] == 1,
                            color='orange', alpha=0.15, label='GT Anomaly')
            ax.set_ylim(ymin, ymax)

        ax.set_ylabel(SENSOR_NAMES[ch], fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='upper right', ncol=3)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=7)

    axes[-1].set_xlabel('Time Step', fontsize=10)
    fig.tight_layout(rect=[0, 0.01, 1, 0.99])

    save_dir = './figures_PSM'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'PSM_CUSUM_AllChannels_DF{DISCOUNT_FACTOR}_b0.5.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {save_path}")
    plt.show()


def main():
    print("=" * 60)
    print("PSM Anomaly Detection - Per-Channel Discount CUSUM")
    print("=" * 60)
    print(f"  SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN}, DISCOUNT={DISCOUNT_FACTOR}")
    print(f"  Channels: {NUM_CHANNELS} ({', '.join(SENSOR_NAMES[:5])}...)")
    print(f"  Split: Train [0,{TRAIN_END}), Val [{TRAIN_END},{VAL_END}), Test [{VAL_END},end)")
    print()

    base_path = find_result_path(SEQ_LEN, PRED_LEN)
    if base_path is None:
        return
    print(f"Result path: {base_path}")

    trues, preds, val_trues, val_preds = load_data(base_path)
    if trues is None or val_trues is None:
        return

    gt_labels = load_ground_truth_labels()
    gt_labels_test = gt_labels[VAL_END:] if gt_labels is not None else None
    if gt_labels_test is not None:
        print(f"Ground Truth: {np.sum(gt_labels_test)} anomaly points / {len(gt_labels_test)} total")

    # =========================================================================
    # Per-channel CUSUM
    # =========================================================================
    all_results = []

    print(f"\n{'=' * 82}")
    print(f"{'Sensor':<10} {'Target':>10} {'Sigma':>10} {'b':>10} {'UCL':>12} {'LCL':>12} {'Warn_UP':>8} {'Warn_DN':>8}")
    print(f"{'=' * 82}")

    for ch in range(NUM_CHANNELS):
        print(f"  Processing {SENSOR_NAMES[ch]}...", end='', flush=True)

        # Phase 1: Validation → calibrate
        res_val = compute_reconstructed_error_vectorized(
            val_trues, val_preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
        valid_err = res_val['signed_error'][~np.isnan(res_val['signed_error'])]

        if len(valid_err) == 0:
            print(f" SKIPPED (no valid error)")
            all_results.append(None)
            continue

        target_mean = np.mean(valid_err)
        sigma = np.std(valid_err)
        slack_b = sigma * 1.0

        temp_up = cusum(valid_err, target_mean, slack_b, 'up')
        temp_down = cusum(valid_err, target_mean, slack_b, 'down')
        ucl = np.max(temp_up)
        lcl = np.min(temp_down)

        # Phase 2: Test → detect
        res_test = compute_reconstructed_error_vectorized(
            trues, preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)

        final_up = cusum(res_test['signed_error'], target_mean, slack_b, 'up')
        final_down = cusum(res_test['signed_error'], target_mean, slack_b, 'down')

        n_up = int(np.sum(final_up > ucl))
        n_dn = int(np.sum(final_down < lcl))

        print(f"\r  {SENSOR_NAMES[ch]:<10} {target_mean:>10.6f} {sigma:>10.6f} {slack_b:>10.6f} {ucl:>12.6f} {lcl:>12.6f} {n_up:>8} {n_dn:>8}")

        all_results.append({
            'channel': ch, 'sensor': SENSOR_NAMES[ch],
            'time_steps': res_test['time_steps'],
            'actual': res_test['actual'], 'pred': res_test['pred'],
            'signed_error': res_test['signed_error'],
            'cusum_up': final_up, 'cusum_down': final_down,
            'ucl': ucl, 'lcl': lcl,
            'target': target_mean, 'sigma': sigma, 'b': slack_b,
        })

    print(f"{'=' * 82}")

    valid_results = [r for r in all_results if r is not None]
    if not valid_results:
        print("Error: No valid channel results.")
        return

    total_up = sum(int(np.sum(r['cusum_up'] > r['ucl'])) for r in valid_results)
    total_dn = sum(int(np.sum(r['cusum_down'] < r['lcl'])) for r in valid_results)
    print(f"\nTotal warnings: {total_up + total_dn} (UP: {total_up}, DOWN: {total_dn})")

    plot_all_channels(all_results, gt_labels_test)


if __name__ == "__main__":
    main()
