"""Daphnet CUSUM Anomaly Detection Animation"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.animation as animation, os, sys, time as time_module

SEQ_LEN = 96; PRED_LEN = 96; DISCOUNT_FACTOR = 0.9
SENSOR_NAMES = ['ankle_horiz_fwd','ankle_vert','ankle_horiz_lateral','leg_horiz_fwd','leg_vert','leg_horiz_lateral','trunk_horiz_fwd','trunk_vert','trunk_horiz_lateral']
NUM_CHANNELS = len(SENSOR_NAMES)
CSV_PATH = './dataset/TSB-AD-M/018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv'
TRAIN_END = 9694; VAL_END = 15001; DATASET_TAG = 'Daphnet'
CHANNEL_TO_SHOW = 0; ANOM_FOCUS_MARGIN = 300; FRAME_STEP = 5; INTERVAL_MS = 50; DPI = 100
FOCUS_CENTER = None; FOCUS_START = None; FOCUS_END = None

def find_result_path(seq_len, pred_len):
    results_dir = './results/'; candidates = [f for f in os.listdir(results_dir) if DATASET_TAG in f and f'seq{seq_len}' in f and f'pred{pred_len}' in f]
    if not candidates: raise FileNotFoundError(f"{DATASET_TAG} result folder not found")
    return os.path.join(results_dir, sorted(candidates)[-1])

def compute_error_and_cusum_streaming(trues, preds, channel, seq_len, pred_len, discount_factor, target, slack_b):
    n_samples = preds.shape[0]; total_len = n_samples + seq_len + pred_len; weights = discount_factor ** np.arange(pred_len)
    pred_ch = preds[:, :, channel]; true_ch = trues[:, :, channel]
    signed_sum = np.zeros(total_len); pred_sum = np.zeros(total_len); true_sum = np.zeros(total_len); weight_sum = np.zeros(total_len)
    print(f"    Computing weighted errors ({n_samples} samples)..."); t_start = time_module.time(); errors = pred_ch - true_ch
    for j in range(pred_len):
        t_indices = np.arange(n_samples) + seq_len + j; valid = t_indices < total_len; t_valid = t_indices[valid]
        np.add.at(signed_sum, t_valid, errors[valid, j] * weights[j]); np.add.at(pred_sum, t_valid, pred_ch[valid, j] * weights[j])
        np.add.at(true_sum, t_valid, true_ch[valid, j] * weights[j]); np.add.at(weight_sum, t_valid, weights[j])
    valid_mask = weight_sum > 0; max_idx = np.where(valid_mask)[0][-1] if np.any(valid_mask) else 0; final_len = max_idx + 1
    error_ts = np.full(final_len, np.nan); pred_ts = np.full(final_len, np.nan); actual_ts = np.full(final_len, np.nan)
    vs = valid_mask[:final_len]; error_ts[vs] = signed_sum[:final_len][vs] / weight_sum[:final_len][vs]
    pred_ts[vs] = pred_sum[:final_len][vs] / weight_sum[:final_len][vs]; actual_ts[vs] = true_sum[:final_len][vs] / weight_sum[:final_len][vs]
    cusum_up = np.zeros(final_len); cusum_down = np.zeros(final_len)
    for t in range(1, final_len):
        if np.isnan(error_ts[t]): cusum_up[t] = cusum_up[t-1]; cusum_down[t] = cusum_down[t-1]
        else: cusum_up[t] = max(0, cusum_up[t-1] + (error_ts[t] - target - slack_b)); cusum_down[t] = min(0, cusum_down[t-1] + (error_ts[t] - target + slack_b))
    print(f"    Done ({time_module.time()-t_start:.1f}s)"); return error_ts, actual_ts, pred_ts, cusum_up, cusum_down

def compute_val_params(val_trues, val_preds, channel, seq_len, pred_len, discount_factor):
    n_samples = val_preds.shape[0]; total_len = n_samples + seq_len + pred_len; weights = discount_factor ** np.arange(pred_len)
    errors = val_preds[:, :, channel] - val_trues[:, :, channel]; signed_sum = np.zeros(total_len); weight_sum = np.zeros(total_len)
    for j in range(pred_len):
        t_indices = np.arange(n_samples) + seq_len + j; valid = t_indices < total_len; t_valid = t_indices[valid]
        np.add.at(signed_sum, t_valid, errors[valid, j] * weights[j]); np.add.at(weight_sum, t_valid, weights[j])
    valid_mask = weight_sum > 0; avg_err = signed_sum[valid_mask] / weight_sum[valid_mask]
    target_mean = np.mean(avg_err); sigma = np.std(avg_err); slack_b = sigma * 1.0
    S_up = np.zeros(len(avg_err)); S_down = np.zeros(len(avg_err))
    for t in range(1, len(avg_err)): S_up[t] = max(0, S_up[t-1] + (avg_err[t] - target_mean - slack_b)); S_down[t] = min(0, S_down[t-1] + (avg_err[t] - target_mean + slack_b))
    return target_mean, sigma, slack_b, np.max(S_up), np.min(S_down)

def find_anomaly_focus_range(gt_labels_test, cusum_up, ucl, margin=200, manual_center=None, manual_start=None, manual_end=None):
    if manual_start is not None and manual_end is not None: start = max(0, manual_start); end = min(len(cusum_up), manual_end); return start, end, (start+end)//2
    if manual_center is not None: start = max(0, manual_center - margin); end = min(len(cusum_up), manual_center + margin); return start, end, manual_center
    gt_start = None; anom_idx = None
    if gt_labels_test is not None: anom_idx = np.where(gt_labels_test == 1)[0]; gt_start = anom_idx[0] if len(anom_idx) > 0 else None
    ucl_breach = np.where(cusum_up > ucl)[0]; cusum_start = ucl_breach[0] if len(ucl_breach) > 0 else None
    candidates = [x for x in [gt_start, cusum_start] if x is not None]
    if not candidates: return 0, min(1000, len(cusum_up)), 500
    focus_point = min(candidates)
    if anom_idx is not None and len(anom_idx) > 0: diffs = np.diff(anom_idx); breaks = np.where(diffs > 1)[0]; anom_end = anom_idx[breaks[0]] if len(breaks) > 0 else anom_idx[-1]
    else: anom_end = focus_point + 300
    return max(0, focus_point - margin), min(len(cusum_up), anom_end + margin), focus_point

def main():
    print("=" * 60); print(f"{DATASET_TAG} CUSUM Animation Generator"); print("=" * 60)
    ch = CHANNEL_TO_SHOW; sensor_name = SENSOR_NAMES[ch]
    print(f"  Sensor: {sensor_name} (channel {ch})"); print(f"  SEQ_LEN={SEQ_LEN}, PRED_LEN={PRED_LEN}, DISCOUNT={DISCOUNT_FACTOR}")
    base_path = find_result_path(SEQ_LEN, PRED_LEN); print(f"  Result path: {base_path}")
    t0 = time_module.time(); trues = np.load(os.path.join(base_path, 'true.npy')); preds = np.load(os.path.join(base_path, 'pred.npy'))
    val_trues = np.load(os.path.join(base_path, 'val_true.npy')); val_preds = np.load(os.path.join(base_path, 'val_pred.npy'))
    print(f"  Test: {trues.shape}, Val: {val_trues.shape}"); n_samples = preds.shape[0]
    try: df = pd.read_csv(CSV_PATH); gt_labels_test = df['Label'].values[VAL_END:]; print(f"  GT anomaly: {np.sum(gt_labels_test)} / {len(gt_labels_test)}")
    except: gt_labels_test = None
    print(f"\n  [Phase 1] Calibrating..."); target_mean, sigma, slack_b, ucl, lcl = compute_val_params(val_trues, val_preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
    print(f"    target={target_mean:.6f}, sigma={sigma:.6f}, UCL={ucl:.6f}, LCL={lcl:.6f}")
    print(f"\n  [Phase 2] Computing CUSUM..."); test_error, test_actual, test_pred, cusum_up_full, cusum_down_full = compute_error_and_cusum_streaming(trues, preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR, target_mean, slack_b)
    focus_start, focus_end, focus_point = find_anomaly_focus_range(gt_labels_test, cusum_up_full, ucl, ANOM_FOCUS_MARGIN, FOCUS_CENTER, FOCUS_START, FOCUS_END)
    print(f"  Focus: [{focus_start}, {focus_end}]")
    frame_indices = list(range(focus_start, focus_end, FRAME_STEP)); n_frames = len(frame_indices); print(f"  Frames: {n_frames}")
    x_margin = 30; x_lo = max(0, focus_start - SEQ_LEN - x_margin); x_hi = min(len(test_actual), focus_end + PRED_LEN + x_margin)
    fig, (ax_ts, ax_cusum) = plt.subplots(2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [1.2, 1]})
    fig.suptitle(f'{DATASET_TAG} iTransformer — {sensor_name} | CUSUM Anomaly Detection', fontsize=14, fontweight='bold')
    bg_x = np.arange(x_lo, x_hi); ax_ts.plot(bg_x, test_actual[x_lo:x_hi], color='#888888', linewidth=0.6, alpha=0.5, label='Reconstructed Actual')
    line_actual, = ax_ts.plot([], [], color='#2196F3', linewidth=1.8, label='Lookback'); line_pred, = ax_ts.plot([], [], color='#F44336', linewidth=1.8, label='Prediction', alpha=0.9)
    line_pred_true, = ax_ts.plot([], [], color='#4CAF50', linewidth=1.2, label='Actual (pred)', linestyle='--', alpha=0.8)
    vline_ts = ax_ts.axvline(x=0, color='#9C27B0', linestyle=':', linewidth=1, alpha=0.7)
    actual_sub = test_actual[x_lo:x_hi]; valid_actual = actual_sub[~np.isnan(actual_sub)]
    if len(valid_actual) > 0: y_pad = (np.max(valid_actual) - np.min(valid_actual)) * 0.15; ax_ts.set_ylim(np.min(valid_actual) - y_pad, np.max(valid_actual) + y_pad)
    ax_ts.set_xlim(x_lo, x_hi); ax_ts.set_ylabel(f'{sensor_name}', fontsize=11); ax_ts.grid(True, alpha=0.3)
    if gt_labels_test is not None:
        gt_len = min(len(gt_labels_test), x_hi)
        if gt_len > x_lo: gt_x = np.arange(x_lo, gt_len); gt_sub = gt_labels_test[x_lo:gt_len]; ymin, ymax = ax_ts.get_ylim(); ax_ts.fill_between(gt_x, ymin, ymax, where=gt_sub == 1, color='orange', alpha=0.15, label='GT Anomaly', zorder=0); ax_ts.set_ylim(ymin, ymax)
    ax_ts.legend(loc='upper left', fontsize=7, ncol=6)
    info_text = ax_ts.text(0.98, 0.95, '', transform=ax_ts.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.4', fc='#FFFDE7', ec='#FFC107', alpha=0.9), family='monospace')
    line_cup, = ax_cusum.plot([], [], color='#F44336', linewidth=1.2, label='CUSUM Up'); line_cdn, = ax_cusum.plot([], [], color='#2196F3', linewidth=1.2, label='CUSUM Down')
    ax_cusum.axhline(y=ucl, color='#F44336', linestyle='--', linewidth=1, alpha=0.7, label=f'UCL: {ucl:.4f}')
    ax_cusum.axhline(y=lcl, color='#2196F3', linestyle='--', linewidth=1, alpha=0.7, label=f'LCL: {lcl:.4f}')
    ax_cusum.axhline(y=0, color='black', linestyle='-', linewidth=0.3); vline_cusum = ax_cusum.axvline(x=0, color='#9C27B0', linestyle=':', linewidth=1, alpha=0.7)
    cup_sub = cusum_up_full[focus_start:focus_end]; cdn_sub = cusum_down_full[focus_start:focus_end]
    cusum_ymax = max(np.nanmax(cup_sub)*1.3, ucl*1.5) if np.nanmax(cup_sub) > 0 else ucl*2; cusum_ymin = min(np.nanmin(cdn_sub)*1.3, lcl*1.5) if np.nanmin(cdn_sub) < 0 else lcl*2
    ax_cusum.set_ylim(cusum_ymin, cusum_ymax); ax_cusum.set_xlim(x_lo, x_hi); ax_cusum.set_xlabel('Time Step (Test)'); ax_cusum.set_ylabel('CUSUM Statistic')
    ax_cusum.legend(loc='upper left', fontsize=7, ncol=4); ax_cusum.grid(True, alpha=0.3)
    if gt_labels_test is not None:
        gt_len = min(len(gt_labels_test), x_hi)
        if gt_len > x_lo: gt_x = np.arange(x_lo, gt_len); gt_sub = gt_labels_test[x_lo:gt_len]; ymin_c, ymax_c = ax_cusum.get_ylim(); ax_cusum.fill_between(gt_x, ymin_c, ymax_c, where=gt_sub == 1, color='orange', alpha=0.1, zorder=0); ax_cusum.set_ylim(ymin_c, ymax_c)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    def update(frame_num):
        t = frame_indices[frame_num]; sample_idx = max(0, t - SEQ_LEN)
        if sample_idx >= n_samples: sample_idx = n_samples - 1
        lb_x = np.arange(sample_idx, min(sample_idx + SEQ_LEN, len(test_actual))); line_actual.set_data(lb_x, test_actual[lb_x[0]:lb_x[-1]+1])
        pred_start_time = sample_idx + SEQ_LEN
        if sample_idx < n_samples:
            pred_x = np.arange(pred_start_time, pred_start_time + PRED_LEN); vr = pred_x < x_hi
            line_pred.set_data(pred_x[vr], preds[sample_idx, :, ch][vr]); line_pred_true.set_data(pred_x[vr], trues[sample_idx, :, ch][vr])
        else: line_pred.set_data([], []); line_pred_true.set_data([], [])
        vline_ts.set_xdata([pred_start_time])
        show_start = max(x_lo, SEQ_LEN); cusum_x = np.arange(show_start, t + 1)
        line_cup.set_data(cusum_x, cusum_up_full[show_start:t+1]); line_cdn.set_data(cusum_x, cusum_down_full[show_start:t+1]); vline_cusum.set_xdata([t])
        err_val = test_error[t] if t < len(test_error) and not np.isnan(test_error[t]) else 0
        cup_val = cusum_up_full[t] if t < len(cusum_up_full) else 0; cdn_val = cusum_down_full[t] if t < len(cusum_down_full) else 0
        info_text.set_text(f"Time: {t}\nError: {err_val:+.5f}\nCUSUM↑: {cup_val:.5f}\nCUSUM↓: {cdn_val:.5f}")
        return (line_actual, line_pred, line_pred_true, vline_ts, line_cup, line_cdn, vline_cusum, info_text)
    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=INTERVAL_MS, blit=True, repeat=False)
    save_dir = f'./figures_{DATASET_TAG}'; os.makedirs(save_dir, exist_ok=True)
    if FOCUS_START is not None and FOCUS_END is not None: focus_tag = f'_range_{FOCUS_START}_{FOCUS_END}'
    else: focus_tag = f'_focus{focus_point}' if FOCUS_CENTER is not None else '_auto'
    gif_path = os.path.join(save_dir, f'{DATASET_TAG}_CUSUM_Animation_{sensor_name}_DF{DISCOUNT_FACTOR}{focus_tag}.gif')
    print(f"  Saving GIF: {gif_path}"); ani.save(gif_path, writer='pillow', dpi=DPI); print(f"  ✓ GIF saved!")
    try: mp4_path = gif_path.replace('.gif', '.mp4'); ani.save(mp4_path, writer='ffmpeg', dpi=DPI, fps=max(1, 1000//INTERVAL_MS)); print(f"  ✓ MP4: {mp4_path}")
    except Exception as e: print(f"  (MP4 skipped: {e})")
    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv) == 2: FOCUS_CENTER = int(sys.argv[1])
    elif len(sys.argv) >= 3: FOCUS_START = int(sys.argv[1]); FOCUS_END = int(sys.argv[2])
    main()
