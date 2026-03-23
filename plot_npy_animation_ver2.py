
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === 설정 ===
base_path = './results/ECL_96_192_iTransformer_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/'  # 결과 폴더 경로로 수정
trues = np.load(base_path + 'true.npy')
preds = np.load(base_path + 'pred.npy')

# --- 파라미터 설정 ---
lookback = 96   # 입력 윈도우 길이 (ex: 48)
pred_len = 192   # 예측 길이 (ex: 96)
channel = 0     # 시각화할 채널

# --- 전체 ground truth 재구성 ---
# (trues: [batch, pred_len, channel])
num_batches, window_size, num_channels = trues.shape
gt = [trues[0, :, channel]]
for i in range(1, num_batches):
    gt.append(trues[i, -1:, channel])
ground_truth = np.concatenate(gt)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(ground_truth, color='gray', label='Ground Truth', alpha=0.7)

# lookback/prediction window 라인
line_lb, = ax.plot([], [], color='blue', label='Lookback')
line_pred, = ax.plot([], [], color='red', label='Prediction')

ax.set_xlim(0, len(ground_truth))
ax.set_ylim(ground_truth.min(), ground_truth.max())
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.legend(loc='upper left')
title = ax.set_title('')

metrics_text = ax.text(0.98, 0.02, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='bottom', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

def update(frame):
    # frame: 예측 시작점 (슬라이딩)
    # 1. lookback window
    lb_start = max(0, frame - lookback)
    lb_end = frame
    lb_x = np.arange(lb_start, lb_end)
    lb_y = ground_truth[lb_start:lb_end]
    line_lb.set_data(lb_x, lb_y)

    # 2. prediction window (예측 결과는 preds에서 추출)
    # preds: [batch, pred_len, channel]
    # frame이 속한 batch와 batch 내 offset 계산
    batch_idx = frame // pred_len
    offset = frame % pred_len
    if batch_idx >= preds.shape[0]:
        # 범위 초과 시 빈 값
        line_pred.set_data([], [])
        metrics_text.set_text('')
        title.set_text('')
        return line_lb, line_pred, title, metrics_text
    pred_y = preds[batch_idx, :, channel]
    pred_x = np.arange(frame, frame + pred_len)
    line_pred.set_data(pred_x, pred_y)

    # ground truth와 예측 구간이 겹치는 부분만 비교
    gt_y = ground_truth[frame:frame + pred_len]
    min_len = min(len(gt_y), len(pred_y))
    mse = np.mean((gt_y[:min_len] - pred_y[:min_len]) ** 2)
    mae = np.mean(np.abs(gt_y[:min_len] - pred_y[:min_len]))
    metrics_text.set_text(f'MSE: {mse:.4f}\nMAE: {mae:.4f}')
    title.set_text(f'Frame: {frame} (Batch {batch_idx})')
    return line_lb, line_pred, title, metrics_text

total_frames = (num_batches - 1) * pred_len
ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=100, blit=True)
ani.save('npy_result_sliding_animation.gif', writer='pillow', dpi=100)
plt.tight_layout()
plt.show()