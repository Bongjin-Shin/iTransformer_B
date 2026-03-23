"""CreditCard CUSUM Anomaly Detection Animation"""
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.animation as animation, os, sys, time as time_module

SEQ_LEN = 96; PRED_LEN = 96; DISCOUNT_FACTOR = 0.9
SENSOR_NAMES = [f'V{i}' for i in range(1, 29)] + ['Amount']
NUM_CHANNELS = len(SENSOR_NAMES)
CSV_PATH = './dataset/TSB-AD-M/137_CreditCard_id_1_Finance_tr_500_1st_541.csv'
TRAIN_END = 4921; VAL_END = 5921; DATASET_TAG = 'CreditCard'
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
    if manual_start is not None and manual_end is not None: return max(0, manual_start), min(len(cusum_up), manual_end), (manual_start+manual_end)//2
    if manual_center is not None: return max(0, manual_center-margin), min(len(cusum_up), manual_center+margin), manual_center
    gt_start = None; anom_idx = None
    if gt_labels_test is not None: anom_idx = np.where(gt_labels_test == 1)[0]; gt_start = anom_idx[0] if len(anom_idx)>0 else None
    ucl_breach = np.where(cusum_up > ucl)[0]; cusum_start = ucl_breach[0] if len(ucl_breach)>0 else None
    candidates = [x for x in [gt_start, cusum_start] if x is not None]
    if not candidates: return 0, min(1000, len(cusum_up)), 500
    fp = min(candidates)
    if anom_idx is not None and len(anom_idx)>0: d=np.diff(anom_idx); b=np.where(d>1)[0]; ae=anom_idx[b[0]] if len(b)>0 else anom_idx[-1]
    else: ae=fp+300
    return max(0,fp-margin), min(len(cusum_up),ae+margin), fp

def main():
    print("=" * 60); print(f"{DATASET_TAG} CUSUM Animation Generator"); print("=" * 60)
    ch = CHANNEL_TO_SHOW; sensor_name = SENSOR_NAMES[ch]
    print(f"  Sensor: {sensor_name} (ch {ch})"); print(f"  SEQ={SEQ_LEN}, PRED={PRED_LEN}, DF={DISCOUNT_FACTOR}")
    base_path = find_result_path(SEQ_LEN, PRED_LEN); print(f"  Path: {base_path}")
    trues = np.load(os.path.join(base_path, 'true.npy')); preds = np.load(os.path.join(base_path, 'pred.npy'))
    val_trues = np.load(os.path.join(base_path, 'val_true.npy')); val_preds = np.load(os.path.join(base_path, 'val_pred.npy'))
    print(f"  Test: {trues.shape}, Val: {val_trues.shape}"); n_samples = preds.shape[0]
    try: df = pd.read_csv(CSV_PATH); gt_labels_test = df['Label'].values[VAL_END:]; print(f"  GT anomaly: {np.sum(gt_labels_test)}/{len(gt_labels_test)}")
    except: gt_labels_test = None
    print(f"\n  [Phase 1] Calibrating..."); target_mean, sigma, slack_b, ucl, lcl = compute_val_params(val_trues, val_preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR)
    print(f"    target={target_mean:.6f}, UCL={ucl:.6f}, LCL={lcl:.6f}")
    print(f"\n  [Phase 2] Computing CUSUM..."); test_error, test_actual, test_pred, cusum_up_full, cusum_down_full = compute_error_and_cusum_streaming(trues, preds, ch, SEQ_LEN, PRED_LEN, DISCOUNT_FACTOR, target_mean, slack_b)
    focus_start, focus_end, focus_point = find_anomaly_focus_range(gt_labels_test, cusum_up_full, ucl, ANOM_FOCUS_MARGIN, FOCUS_CENTER, FOCUS_START, FOCUS_END)
    frame_indices = list(range(focus_start, focus_end, FRAME_STEP)); n_frames = len(frame_indices)
    x_margin = 30; x_lo = max(0, focus_start-SEQ_LEN-x_margin); x_hi = min(len(test_actual), focus_end+PRED_LEN+x_margin)
    fig, (ax_ts, ax_cusum) = plt.subplots(2, 1, figsize=(16,9), gridspec_kw={'height_ratios':[1.2,1]})
    fig.suptitle(f'{DATASET_TAG} iTransformer — {sensor_name} | CUSUM', fontsize=14, fontweight='bold')
    bg_x = np.arange(x_lo, x_hi); ax_ts.plot(bg_x, test_actual[x_lo:x_hi], color='#888888', linewidth=0.6, alpha=0.5, label='Actual')
    line_actual, = ax_ts.plot([],[],color='#2196F3',linewidth=1.8,label='Lookback'); line_pred, = ax_ts.plot([],[],color='#F44336',linewidth=1.8,label='Prediction',alpha=0.9)
    line_pred_true, = ax_ts.plot([],[],color='#4CAF50',linewidth=1.2,label='Actual(pred)',linestyle='--',alpha=0.8)
    vline_ts = ax_ts.axvline(x=0,color='#9C27B0',linestyle=':',linewidth=1,alpha=0.7)
    actual_sub = test_actual[x_lo:x_hi]; va = actual_sub[~np.isnan(actual_sub)]
    if len(va)>0: yp=(np.max(va)-np.min(va))*0.15; ax_ts.set_ylim(np.min(va)-yp, np.max(va)+yp)
    ax_ts.set_xlim(x_lo,x_hi); ax_ts.set_ylabel(sensor_name); ax_ts.grid(True,alpha=0.3)
    if gt_labels_test is not None:
        gt_len=min(len(gt_labels_test),x_hi)
        if gt_len>x_lo: gt_x=np.arange(x_lo,gt_len); gs=gt_labels_test[x_lo:gt_len]; ym,yM=ax_ts.get_ylim(); ax_ts.fill_between(gt_x,ym,yM,where=gs==1,color='orange',alpha=0.15,label='GT',zorder=0); ax_ts.set_ylim(ym,yM)
    ax_ts.legend(loc='upper left',fontsize=7,ncol=6)
    info_text = ax_ts.text(0.98,0.95,'',transform=ax_ts.transAxes,fontsize=9,va='top',ha='right',bbox=dict(boxstyle='round,pad=0.4',fc='#FFFDE7',ec='#FFC107',alpha=0.9),family='monospace')
    line_cup, = ax_cusum.plot([],[],color='#F44336',linewidth=1.2,label='CUSUM Up'); line_cdn, = ax_cusum.plot([],[],color='#2196F3',linewidth=1.2,label='CUSUM Down')
    ax_cusum.axhline(y=ucl,color='#F44336',linestyle='--',linewidth=1,alpha=0.7,label=f'UCL:{ucl:.4f}')
    ax_cusum.axhline(y=lcl,color='#2196F3',linestyle='--',linewidth=1,alpha=0.7,label=f'LCL:{lcl:.4f}')
    ax_cusum.axhline(y=0,color='black',linestyle='-',linewidth=0.3); vline_cusum = ax_cusum.axvline(x=0,color='#9C27B0',linestyle=':',linewidth=1,alpha=0.7)
    cup_sub=cusum_up_full[focus_start:focus_end]; cdn_sub=cusum_down_full[focus_start:focus_end]
    cymax=max(np.nanmax(cup_sub)*1.3,ucl*1.5) if np.nanmax(cup_sub)>0 else ucl*2; cymin=min(np.nanmin(cdn_sub)*1.3,lcl*1.5) if np.nanmin(cdn_sub)<0 else lcl*2
    ax_cusum.set_ylim(cymin,cymax); ax_cusum.set_xlim(x_lo,x_hi); ax_cusum.set_xlabel('Time Step'); ax_cusum.set_ylabel('CUSUM')
    ax_cusum.legend(loc='upper left',fontsize=7,ncol=4); ax_cusum.grid(True,alpha=0.3)
    if gt_labels_test is not None:
        gt_len=min(len(gt_labels_test),x_hi)
        if gt_len>x_lo: gt_x=np.arange(x_lo,gt_len); gs=gt_labels_test[x_lo:gt_len]; yc,yC=ax_cusum.get_ylim(); ax_cusum.fill_between(gt_x,yc,yC,where=gs==1,color='orange',alpha=0.1,zorder=0); ax_cusum.set_ylim(yc,yC)
    fig.tight_layout(rect=[0,0,1,0.96])
    def update(frame_num):
        t=frame_indices[frame_num]; si=max(0,t-SEQ_LEN)
        if si>=n_samples: si=n_samples-1
        lb_x=np.arange(si,min(si+SEQ_LEN,len(test_actual))); line_actual.set_data(lb_x,test_actual[lb_x[0]:lb_x[-1]+1])
        pst=si+SEQ_LEN
        if si<n_samples: px=np.arange(pst,pst+PRED_LEN); vr=px<x_hi; line_pred.set_data(px[vr],preds[si,:,ch][vr]); line_pred_true.set_data(px[vr],trues[si,:,ch][vr])
        else: line_pred.set_data([],[]); line_pred_true.set_data([],[])
        vline_ts.set_xdata([pst]); ss=max(x_lo,SEQ_LEN); cx=np.arange(ss,t+1)
        line_cup.set_data(cx,cusum_up_full[ss:t+1]); line_cdn.set_data(cx,cusum_down_full[ss:t+1]); vline_cusum.set_xdata([t])
        ev=test_error[t] if t<len(test_error) and not np.isnan(test_error[t]) else 0
        cu=cusum_up_full[t] if t<len(cusum_up_full) else 0; cd=cusum_down_full[t] if t<len(cusum_down_full) else 0
        info_text.set_text(f"Time:{t}\nErr:{ev:+.5f}\nCU↑:{cu:.5f}\nCU↓:{cd:.5f}")
        return (line_actual,line_pred,line_pred_true,vline_ts,line_cup,line_cdn,vline_cusum,info_text)
    ani = animation.FuncAnimation(fig,update,frames=n_frames,interval=INTERVAL_MS,blit=True,repeat=False)
    save_dir=f'./figures_{DATASET_TAG}'; os.makedirs(save_dir,exist_ok=True)
    if FOCUS_START is not None and FOCUS_END is not None: ft=f'_range_{FOCUS_START}_{FOCUS_END}'
    else: ft=f'_focus{focus_point}' if FOCUS_CENTER is not None else '_auto'
    gp=os.path.join(save_dir,f'{DATASET_TAG}_CUSUM_Animation_{sensor_name}_DF{DISCOUNT_FACTOR}{ft}.gif')
    print(f"  Saving: {gp}"); ani.save(gp,writer='pillow',dpi=DPI); print("  ✓ GIF saved!")
    try: mp=gp.replace('.gif','.mp4'); ani.save(mp,writer='ffmpeg',dpi=DPI,fps=max(1,1000//INTERVAL_MS)); print(f"  ✓ MP4: {mp}")
    except Exception as e: print(f"  (MP4 skipped: {e})")
    print("\nDone!")

if __name__ == "__main__":
    if len(sys.argv)==2: FOCUS_CENTER=int(sys.argv[1])
    elif len(sys.argv)>=3: FOCUS_START=int(sys.argv[1]); FOCUS_END=int(sys.argv[2])
    main()
