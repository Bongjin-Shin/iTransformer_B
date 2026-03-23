"""PSM CUSUM Anomaly Detection Animation
iTransformer의 예측과 CUSUM 이상탐지가 실시간으로 진행되는 과정을 시각화.
급격한 anomaly 발생 시점 부근에 집중한 animation을 생성.

구성:
  - 상단: 실제값 vs 예측값 (lookback + prediction window 슬라이딩)
  - 하단: CUSUM Up/Down + UCL/LCL 경계선 실시간 업데이트

성능 최적화:
  - Phase 1: 전체 CUSUM을 미리 계산 (progress bar 표시)
  - Phase 2: 계산된 결과에서 Focus 구간만 추출하여 Animation 생성
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for faster rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import time as time_module

# ============== Configuration ==============
SEQ_LEN = 96
PRED_LEN = 96
DISCOUNT_FACTOR = 0.9

SENSOR_NAMES = [f'feature_{i}' for i in range(25)]
NUM_CHANNELS = len(SENSOR_NAMES)

CSV_PATH = './dataset/TSB-AD-M/115_PSM_id_1_Facility_tr_50000_1st_129872.csv'
TRAIN_END = 50001
VAL_END = 60001

# Animation parameters
CHANNEL_TO_SHOW = 0         # 시각화할 센서 인덱스 (0=feature_0, 1=feature_1, ...)
ANOM_FOCUS_MARGIN = 300     # anomaly 발생 전후 여유 프레임
FRAME_STEP = 5              # 프레임 스텝 (매 N번째 시점을 프레임으로 사용, 클수록 빠름)
INTERVAL_MS = 50            # 프레임 간격 (ms, 작을수록 빠름)
DPI = 100

# 수동 Focus 구간 설정 (None이면 자동 탐지)
FOCUS_CENTER = None
FOCUS_START = None
FOCUS_END = None
# ==================================================


def find_result_path(seq_len, pred_len):
    results_dir = './results/'
    candidates = [f for f in os.listdir(results_dir)
                  if 'PSM' in f and f'seq{seq_len}' in f and f'pred{pred_len}' in f]
    if not candidates:
        raise FileNotFoundError("PSM result folder not found")
    return os.path.join(results_dir, sorted(candidates)[-1])


def compute_error_and_cusum_streaming(trues, preds, channel, seq_len, pred_len,
                                       discount_factor, target, slack_b):
    """
    Streaming 방식으로 discount-weighted error + CUSUM을 동시에 계산.
    메모리 효율적: 전체 시간축 배열 대신 sliding window로 처리.
    
    Returns:
        error_ts: 시간축별 weighted signed error (length = n_samples + seq_len + pred_len)
        actual_ts: 시간축별 actual value
        pred_ts: 시간축별 weighted prediction
        cusum_up: CUSUM up statistic
        cusum_down: CUSUM down statistic
    """
    n_samples = preds.shape[0]
    total_len = n_samples + seq_len + pred_len
    
    weights = discount_factor ** np.arange(pred_len)
    
    # Pre-extract channel data
    pred_ch = preds[:, :, channel]  # (n_samples, pred_len)
    true_ch = trues[:, :, channel]
    
    # Accumulation arrays
    signed_sum = np.zeros(total_len)
    pred_sum = np.zeros(total_len)
    true_sum = np.zeros(total_len)
    weight_sum = np.zeros(total_len)
    
    print(f"    Computing weighted errors ({n_samples} samples, {pred_len} horizons)...")
    t_start = time_module.time()
    
    # Vectorized accumulation per horizon step
    errors = pred_ch - true_ch  # (n_samples, pred_len)
    for j in range(pred_len):
        t_indices = np.arange(n_samples) + seq_len + j
        valid = t_indices < total_len
        t_valid = t_indices[valid]
        np.add.at(signed_sum, t_valid, errors[valid, j] * weights[j])
        np.add.at(pred_sum, t_valid, pred_ch[valid, j] * weights[j])
        np.add.at(true_sum, t_valid, true_ch[valid, j] * weights[j])
        np.add.at(weight_sum, t_valid, weights[j])
        
        if (j + 1) % 20 == 0:
            print(f"      horizon {j+1}/{pred_len} done ({time_module.time()-t_start:.1f}s)")
    
    print(f"    Accumulation done ({time_module.time()-t_start:.1f}s)")
    
    # Compute averages
    valid_mask = weight_sum > 0
    max_idx = np.where(valid_mask)[0][-1] if np.any(valid_mask) else 0
    final_len = max_idx + 1
    
    error_ts = np.full(final_len, np.nan)
    pred_ts = np.full(final_len, np.nan)
    actual_ts = np.full(final_len, np.nan)
    
    vs = valid_mask[:final_len]
    error_ts[vs] = signed_sum[:final_len][vs] / weight_sum[:final_len][vs]
    pred_ts[vs] = pred_sum[:final_len][vs] / weight_sum[:final_len][vs]
    actual_ts[vs] = true_sum[:final_len][vs] / weight_sum[:final_len][vs]
    
    # CUSUM computation
    print(f"    Computing CUSUM ({final_len} time steps)...")
    cusum_up = np.zeros(final_len)
    cusum_down = np.zeros(final_len)
    for t in range(1, final_len):
        if np.isnan(error_ts[t]):
            cusum_up[t] = cusum_up[t-1]
            cusum_down[t] = cusum_down[t-1]
        else:
            cusum_up[t] = max(0, cusum_up[t-1] + (error_ts[t] - target - slack_b))
            cusum_down[t] = min(0, cusum_down[t-1] + (error_ts[t] - target + slack_b))
    
    print(f"    Total computation: {time_module.time()-t_start:.1f}s")
    
    return error_ts, actual_ts, pred_ts, cusum_up, cusum_down


def compute_val_params(val_trues, val_preds, channel, seq_len, pred_len, discount_factor):
    """Validation 데이터에서 CUSUM 파라미터 추출 (작은 데이터, 빠름)"""
    n_samples = val_preds.shape[0]
    total_len = n_samples + seq_len + pred_len
    
    weights = discount_factor ** np.arange(pred_len)
    errors = val_preds[:, :, channel] - val_trues[:, :, channel]
    
    signed_sum = np.zeros(total_len)
    weight_sum = np.zeros(total_len)
    
    for j in range(pred_len):
        t_indices = np.arange(n_samples) + seq_len + j
        valid = t_indices < total_len
        t_valid = t_indices[valid]
        np.add.at(signed_sum, t_valid, errors[valid, j] * weights[j])
        np.add.at(weight_sum, t_valid, weights[j])
    
    valid_mask = weight_sum > 0
    avg_err = signed_sum[valid_mask] / weight_sum[valid_mask]
    
    target_mean = np.mean(avg_err)
    sigma = np.std(avg_err)
    slack_b = sigma * 1.0
    
    # CUSUM on validation errors for UCL/LCL
    S_up = np.zeros(len(avg_err))
    S_down = np.zeros(len(avg_err))
    for t in range(1, len(avg_err)):
        S_up[t] = max(0, S_up[t-1] + (avg_err[t] - target_mean - slack_b))
        S_down[t] = min(0, S_down[t-1] + (avg_err[t] - target_mean + slack_b))
    
    ucl = np.max(S_up)
    lcl = np.min(S_down)
    
    return target_mean, sigma, slack_b, ucl, lcl


def find_anomaly_focus_range(gt_labels_test, cusum_up, ucl, margin=200, manual_center=None, manual_start=None, manual_end=None):
    """Focus range around the anomaly onset."""
    # Manual override for full range
    if manual_start is not None and manual_end is not None:
        start = max(0, manual_start)
        end = min(len(cusum_up), manual_end)
        return start, end, (start + end) // 2

    # Manual override for center
    if manual_center is not None:
        start = max(0, manual_center - margin)
        end = min(len(cusum_up), manual_center + margin)
        return start, end, manual_center

    gt_start = None
    anom_idx = None
    if gt_labels_test is not None:
        anom_idx = np.where(gt_labels_test == 1)[0]
        if len(anom_idx) > 0:
            gt_start = anom_idx[0]

    ucl_breach = np.where(cusum_up > ucl)[0]
    cusum_start = ucl_breach[0] if len(ucl_breach) > 0 else None

    candidates = [x for x in [gt_start, cusum_start] if x is not None]
    if not candidates:
        return 0, min(1000, len(cusum_up)), 500
    focus_point = min(candidates)

    if anom_idx is not None and len(anom_idx) > 0:
        diffs = np.diff(anom_idx)
        breaks = np.where(diffs > 1)[0]
        anom_end = anom_idx[breaks[0]] if len(breaks) > 0 else anom_idx[-1]
    else:
        anom_end = focus_point + 300

    start = max(0, focus_point - margin)
    end = min(len(cusum_up), anom_end + margin)
    return start, end, focus_point


def main():
    print("=" * 60)
    print("PSM CUSUM Animation Generator")
    print("=" * 60)

    ch = CHANNEL_TO_SHOW
    sensor_name = SENSOR_NAMES[ch]
    print(f"  Sensor: {sensor_name} (channel {ch})")
    print(f"  SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN}, DISCOUNT={DISCOUNT_FACTOR}")

    # --- Load data ---
    base_path = find_result_path(SEQ_LEN, PRED_LEN)
    print(f"  Result path: {base_path}")

    print("  Loading npy files...")
    t0 = time_module.time()
    trues = np.load(os.path.join(base_path, 'true.npy'))
    preds = np.load(os.path.join(base_path, 'pred.npy'))
    val_trues = np.load(os.path.join(base_path, 'val_true.npy'))
    val_preds = np.load(os.path.join(base_path, 'val_pred.npy'))
    print(f"  Loaded in {time_module.time()-t0:.1f}s")
    print(f"  Test:  trues {trues.shape}, preds {preds.shape}")
    print(f"  Val:   trues {val_trues.shape}, preds {val_preds.shape}")
    n_samples = preds.shape[0]

    # Ground truth labels
    try:
        df = pd.read_csv(CSV_PATH)
        gt_labels_test = df['Label'].values[VAL_END:]
        print(f"  GT anomaly points: {np.sum(gt_labels_test)} / {len(gt_labels_test)}")
    except Exception:
        gt_labels_test = None
        print("  Warning: GT labels not loaded")

    # --- Phase 1: Validation → UCL/LCL calibration ---
    print(f"\n  [Phase 1] Calibrating UCL/LCL from validation set...")
    target_mean, sigma, slack_b, ucl, lcl = compute_val_params(
        val_trues, val_preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
    print(f"    target={target_mean:.6f}, sigma={sigma:.6f}, b={slack_b:.6f}")
    print(f"    UCL={ucl:.6f}, LCL={lcl:.6f}")

    # --- Phase 2: Test → full error & CUSUM ---
    print(f"\n  [Phase 2] Computing test errors & CUSUM...")
    test_error, test_actual, test_pred, cusum_up_full, cusum_down_full = \
        compute_error_and_cusum_streaming(
            trues, preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR,
            target_mean, slack_b)

    # --- Focus range ---
    focus_start, focus_end, focus_point = find_anomaly_focus_range(
        gt_labels_test, cusum_up_full, ucl, ANOM_FOCUS_MARGIN, 
        manual_center=FOCUS_CENTER, manual_start=FOCUS_START, manual_end=FOCUS_END)
    print(f"  Focus range: [{focus_start}, {focus_end}] (center={focus_point})")

    # =====================================================================
    # Animation Setup — 프레임 수 제한으로 메모리 보호
    # =====================================================================
    MAX_FRAMES = 600  # GIF 최대 프레임 수
    focus_span = focus_end - focus_start
    effective_step = FRAME_STEP
    if focus_span // FRAME_STEP > MAX_FRAMES:
        effective_step = max(1, focus_span // MAX_FRAMES)
        print(f"  ⚠ Focus range too large ({focus_span} steps). "
              f"Auto-adjusting FRAME_STEP: {FRAME_STEP} → {effective_step}")
    frame_indices = list(range(focus_start, focus_end, effective_step))
    n_frames = len(frame_indices)
    print(f"  Total animation frames: {n_frames}")

    # Precompute x-axis range
    x_margin = 30
    x_lo = max(0, focus_start - SEQ_LEN - x_margin)
    x_hi = min(len(test_actual), focus_end + PRED_LEN + x_margin)

    fig, (ax_ts, ax_cusum) = plt.subplots(2, 1, figsize=(16, 9),
                                           gridspec_kw={'height_ratios': [1.2, 1]})
    fig.suptitle(f'PSM iTransformer — {sensor_name} Sensor | CUSUM Anomaly Detection',
                 fontsize=14, fontweight='bold')

    # ---- Top: Time Series (Actual vs Prediction) ----
    bg_x = np.arange(x_lo, x_hi)
    ax_ts.plot(bg_x, test_actual[x_lo:x_hi], color='#888888', linewidth=0.6, alpha=0.5,
               label='Reconstructed Actual')

    # Dynamic lines
    line_actual, = ax_ts.plot([], [], color='#2196F3', linewidth=1.8, label='Lookback (actual)')
    line_pred, = ax_ts.plot([], [], color='#F44336', linewidth=1.8, label='Prediction', alpha=0.9)
    line_pred_true, = ax_ts.plot([], [], color='#4CAF50', linewidth=1.2,
                                  label='Actual (pred horizon)', linestyle='--', alpha=0.8)
    vline_ts = ax_ts.axvline(x=0, color='#9C27B0', linestyle=':', linewidth=1, alpha=0.7,
                             label='Pred start')

    # ylim
    actual_sub = test_actual[x_lo:x_hi]
    valid_actual = actual_sub[~np.isnan(actual_sub)]
    if len(valid_actual) > 0:
        y_pad = (np.max(valid_actual) - np.min(valid_actual)) * 0.15
        ax_ts.set_ylim(np.min(valid_actual) - y_pad, np.max(valid_actual) + y_pad)
    ax_ts.set_xlim(x_lo, x_hi)
    ax_ts.set_ylabel(f'{sensor_name} Value', fontsize=11)
    ax_ts.grid(True, alpha=0.3)

    # GT shading (top)
    if gt_labels_test is not None:
        gt_len = min(len(gt_labels_test), x_hi)
        if gt_len > x_lo:
            gt_x = np.arange(x_lo, gt_len)
            gt_sub = gt_labels_test[x_lo:gt_len]
            ymin, ymax = ax_ts.get_ylim()
            ax_ts.fill_between(gt_x, ymin, ymax, where=gt_sub == 1,
                               color='orange', alpha=0.15, label='GT Anomaly', zorder=0)
            ax_ts.set_ylim(ymin, ymax)

    ax_ts.legend(loc='upper left', fontsize=7, ncol=6)

    # Info text
    info_text = ax_ts.text(0.98, 0.95, '', transform=ax_ts.transAxes, fontsize=9,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.4', fc='#FFFDE7',
                                     ec='#FFC107', alpha=0.9), family='monospace')

    # ---- Bottom: CUSUM chart ----
    line_cup, = ax_cusum.plot([], [], color='#F44336', linewidth=1.2, label='CUSUM Up')
    line_cdn, = ax_cusum.plot([], [], color='#2196F3', linewidth=1.2, label='CUSUM Down')
    ax_cusum.axhline(y=ucl, color='#F44336', linestyle='--', linewidth=1, alpha=0.7,
                     label=f'UCL: {ucl:.4f}')
    ax_cusum.axhline(y=lcl, color='#2196F3', linestyle='--', linewidth=1, alpha=0.7,
                     label=f'LCL: {lcl:.4f}')
    ax_cusum.axhline(y=0, color='black', linestyle='-', linewidth=0.3)
    vline_cusum = ax_cusum.axvline(x=0, color='#9C27B0', linestyle=':', linewidth=1, alpha=0.7)

    # CUSUM y-range
    cup_sub = cusum_up_full[focus_start:focus_end]
    cdn_sub = cusum_down_full[focus_start:focus_end]
    cusum_ymax = max(np.nanmax(cup_sub) * 1.3, ucl * 1.5) if np.nanmax(cup_sub) > 0 else ucl * 2
    cusum_ymin = min(np.nanmin(cdn_sub) * 1.3, lcl * 1.5) if np.nanmin(cdn_sub) < 0 else lcl * 2
    ax_cusum.set_ylim(cusum_ymin, cusum_ymax)
    ax_cusum.set_xlim(x_lo, x_hi)
    ax_cusum.set_xlabel('Time Step (Test)', fontsize=11)
    ax_cusum.set_ylabel('CUSUM Statistic', fontsize=11)
    ax_cusum.legend(loc='upper left', fontsize=7, ncol=4)
    ax_cusum.grid(True, alpha=0.3)

    # GT shading (CUSUM)
    if gt_labels_test is not None:
        gt_len = min(len(gt_labels_test), x_hi)
        if gt_len > x_lo:
            gt_x = np.arange(x_lo, gt_len)
            gt_sub = gt_labels_test[x_lo:gt_len]
            ymin_c, ymax_c = ax_cusum.get_ylim()
            ax_cusum.fill_between(gt_x, ymin_c, ymax_c, where=gt_sub == 1,
                                  color='orange', alpha=0.1, zorder=0)
            ax_cusum.set_ylim(ymin_c, ymax_c)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # =====================================================================
    # Animation Update
    # =====================================================================
    def update(frame_num):
        t = frame_indices[frame_num]

        # sample index: sample i's prediction starts at time (i + SEQ_LEN)
        sample_idx = max(0, t - SEQ_LEN)
        if sample_idx >= n_samples:
            sample_idx = n_samples - 1

        # Lookback window (actual values before prediction start)
        lb_start = sample_idx
        lb_end = min(sample_idx + SEQ_LEN, len(test_actual))
        lb_x = np.arange(lb_start, lb_end)
        lb_y = test_actual[lb_start:lb_end]
        line_actual.set_data(lb_x, lb_y)

        # Prediction window from raw npy
        pred_start_time = sample_idx + SEQ_LEN
        if sample_idx < n_samples:
            pred_y_raw = preds[sample_idx, :, ch]
            true_y_raw = trues[sample_idx, :, ch]
            pred_x = np.arange(pred_start_time, pred_start_time + PRED_LEN)
            valid_range = pred_x < x_hi
            line_pred.set_data(pred_x[valid_range], pred_y_raw[valid_range])
            line_pred_true.set_data(pred_x[valid_range], true_y_raw[valid_range])
        else:
            line_pred.set_data([], [])
            line_pred_true.set_data([], [])

        vline_ts.set_xdata([pred_start_time])

        # CUSUM trace up to current time
        show_start = max(x_lo, SEQ_LEN)
        cusum_x = np.arange(show_start, t + 1)
        line_cup.set_data(cusum_x, cusum_up_full[show_start:t + 1])
        line_cdn.set_data(cusum_x, cusum_down_full[show_start:t + 1])
        vline_cusum.set_xdata([t])

        # Info text
        err_val = test_error[t] if t < len(test_error) and not np.isnan(test_error[t]) else 0
        cup_val = cusum_up_full[t] if t < len(cusum_up_full) else 0
        cdn_val = cusum_down_full[t] if t < len(cusum_down_full) else 0
        info_text.set_text(f"Time: {t}\nError: {err_val:+.5f}\n"
                           f"CUSUM↑: {cup_val:.5f}\nCUSUM↓: {cdn_val:.5f}")

        return (line_actual, line_pred, line_pred_true, vline_ts,
                line_cup, line_cdn, vline_cusum, info_text)

    # =====================================================================
    # Save
    # =====================================================================
    print(f"\n  Creating animation ({n_frames} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=INTERVAL_MS, blit=True, repeat=False)

    save_dir = './figures_PSM'
    os.makedirs(save_dir, exist_ok=True)
    if FOCUS_START is not None and FOCUS_END is not None:
        focus_tag = f'_range_{FOCUS_START}_{FOCUS_END}'
    else:
        focus_tag = f'_focus{focus_point}' if FOCUS_CENTER is not None else '_auto'
    
    gif_path = os.path.join(save_dir,
                            f'PSM_CUSUM_Animation_{sensor_name}_DF{DISCOUNT_FACTOR}{focus_tag}.gif')
    print(f"  Saving GIF: {gif_path}")
    ani.save(gif_path, writer='pillow', dpi=DPI)
    print(f"  ✓ GIF saved!")

    try:
        mp4_path = gif_path.replace('.gif', '.mp4')
        ani.save(mp4_path, writer='ffmpeg', dpi=DPI, fps=max(1, 1000 // INTERVAL_MS))
        print(f"  ✓ MP4 saved: {mp4_path}")
    except Exception as e:
        print(f"  (MP4 skipped: {e})")

    print("\nDone!")


if __name__ == "__main__":
    # 커맨드라인 인자 (1개면 center, 2개면 start/end)
    if len(sys.argv) == 2:
        FOCUS_CENTER = int(sys.argv[1])
        print(f"[CLI] FOCUS_CENTER = {FOCUS_CENTER}")
    elif len(sys.argv) >= 3:
        FOCUS_START = int(sys.argv[1])
        FOCUS_END = int(sys.argv[2])
        print(f"[CLI] FOCUS_RANGE = [{FOCUS_START}, {FOCUS_END}]")
    main()
