"""
C-MAPSS Sliding Window Prediction Visualization (Animation)

Visualize test results as GIF animation:
- Ground Truth displayed as background (gray)
- Lookback window (input) in blue
- Prediction in red
- Normal limit line at row 100 (normal/degradation boundary)
- Real-time MSE, MAE metrics display

Usage:
    python visualize_cmapss_sliding_animation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

# ============================================================
# Configuration
# ============================================================
# Direct configuration for seq_len and pred_len
SEQ_LEN = 48      # lookback window size
PRED_LEN = 48     # prediction length

# Sensor names (excluded: s1, s5, s6, s10, s16, s18, s19)
SENSOR_NAMES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# Channel to visualize (0~13)
CHANNEL = 5  # Sensor index to visualize

# Engine index to visualize (0~19, test engines 81~100)
ENGINE_INDEX = 1

# Normal pattern training limit (train_limit)
NORMAL_LIMIT = 100

# Animation settings
FRAME_INTERVAL = 150  # milliseconds (frame interval)
DPI = 100

# Output folder
OUTPUT_DIR = './gif/cmapss_sliding/'


def find_result_path(seq_len, pred_len):
    """Find result folder path with seq_len and pred_len
    
    Folder naming pattern:
    train_FD001_seq{SEQ}_iTransformer_CMAPSS_M_ft{SEQ}_sl{LABEL}_ll{PRED}_pl{D_MODEL}...
    
    - 'seq{SEQ}' or 'ft{SEQ}' contains seq_len
    - 'll{PRED}' contains pred_len
    """
    results_dir = './results/'
    
    if not os.path.exists(results_dir):
        print(f"Error: results folder not found: {results_dir}")
        return None
    
    # Pattern matching: seq_len in model_id, pred_len in 'll{pred_len}'
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
        # Return the most recent (last alphabetically)
        return os.path.join(results_dir, sorted(candidates)[-1])
    
    print(f"Error: Cannot find result folder with seq_len={seq_len}, pred_len={pred_len}")
    print(f"Available CMAPSS folders:")
    for folder in sorted(os.listdir(results_dir)):
        if 'CMAPSS' in folder and 'Progressive' not in folder and 'Expanding' not in folder:
            print(f"  - {folder}")
    return None


def load_data(base_path):
    """Load data"""
    try:
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        print(f"Data loaded successfully: {base_path}")
        print(f"  trues shape: {trues.shape}")
        print(f"  preds shape: {preds.shape}")
        return trues, preds
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return None, None


def get_engine_info():
    """Load test engine information from data"""
    try:
        df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
        test_engine_ids = list(range(81, 101))
        
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


def reconstruct_ground_truth(trues, sample_start, sample_end, channel, seq_len, pred_len):
    """해당 엔진의 Ground Truth 전체 시그널 재구성"""
    engine_trues = trues[sample_start:sample_end]
    num_samples = engine_trues.shape[0]
    
    if num_samples == 0:
        return None
    
    # 첫 번째 샘플의 prediction 부분 + 각 샘플의 마지막 prediction 포인트
    # 실제로는 슬라이딩 윈도우이므로 첫 샘플 전체 + 나머지 마지막 포인트
    
    # 전체 길이: seq_len + pred_len + (num_samples - 1)
    total_length = seq_len + pred_len + (num_samples - 1)
    
    reconstructed = np.zeros(total_length)
    
    # 첫 번째 샘플: lookback은 없으므로 prediction만 사용
    # 하지만 trues는 label_len + pred_len 형태
    # 여기서는 prediction 구간만 사용
    
    # 방법: 각 prediction이 시작되는 위치를 기준으로 재구성
    # sample i의 prediction은 position (seq_len + i) 부터 시작
    
    for i in range(num_samples):
        pred_values = engine_trues[i, :, channel]  # [label_len + pred_len]
        pred_start_pos = seq_len + i  # lookback 끝 + i
        
        # prediction의 마지막 pred_len 부분만 사용
        actual_pred = pred_values[-pred_len:]
        
        for j, val in enumerate(actual_pred):
            pos = pred_start_pos + j
            if pos < total_length:
                reconstructed[pos] = val
    
    # 앞부분 (lookback 구간)은 첫 샘플 기준으로 복원 (불완전)
    # 대신 원본 데이터에서 로드
    return reconstructed, total_length


def load_ground_truth_from_csv(engine_id, sensor_names, channel):
    """원본 CSV에서 Ground Truth 로드"""
    try:
        df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
        engine_data = df[df['id'] == engine_id].reset_index(drop=True)
        
        sensor_col = sensor_names[channel]
        ground_truth = engine_data[sensor_col].values
        
        return ground_truth
    except Exception as e:
        print(f"Ground Truth 로드 실패: {e}")
        return None


def create_animation_for_engine(trues, preds, engine_idx, channel, seq_len, pred_len, 
                                 engine_info, test_engine_ids, output_path):
    """특정 엔진에 대한 애니메이션 생성"""
    
    # 엔진별 샘플 범위 계산
    sample_start, sample_end, engine_id, engine_len = get_sample_range_for_engine(
        engine_idx, engine_info, test_engine_ids, seq_len, pred_len
    )
    
    num_samples = sample_end - sample_start
    print(f"\n엔진 {engine_id} 정보:")
    print(f"  - 총 길이: {engine_len} 행")
    print(f"  - 샘플 범위: {sample_start} ~ {sample_end} (총 {num_samples}개)")
    
    if num_samples <= 0:
        print(f"Error: No samples for engine {engine_id}")
        return
    
    # Ground Truth from test results (scaled)
    engine_trues = trues[sample_start:sample_end]
    engine_preds = preds[sample_start:sample_end]
    
    total_length = engine_len
    
    # =====================================================
    # Ground Truth reconstruction from true.npy
    # Each sample contains [label_len + pred_len] data
    # Sample i: lookback ends at position (i + seq_len)
    #           prediction covers [i + seq_len, i + seq_len + pred_len)
    # =====================================================
    reconstructed_gt = np.zeros(total_length)
    
    # Method: Use overlapping samples to reconstruct full signal
    # For each position, use the first sample that covers it
    
    for i in range(num_samples):
        # Sample i covers prediction range [seq_len + i, seq_len + i + pred_len)
        # But engine_trues[i] has shape [label_len + pred_len] or just [pred_len]
        # Actually, from test output, it's [pred_len] (only prediction part)
        
        sample_true = engine_trues[i, :, channel]  # [pred_len]
        
        # Prediction starts at position seq_len + i
        pred_start = seq_len + i
        
        for j in range(len(sample_true)):
            pos = pred_start + j
            if pos < total_length:
                reconstructed_gt[pos] = sample_true[j]
    
    # For the first seq_len positions (lookback area of first sample),
    # we need to reconstruct from the sliding nature
    # Position p is covered by sample (p - seq_len) if p >= seq_len
    # For p < seq_len, we need to infer from the pattern
    
    # Use first few samples to reconstruct early positions
    # Sample 0's prediction covers [seq_len, seq_len + pred_len)
    # Sample 1's prediction covers [seq_len + 1, seq_len + 1 + pred_len)
    # So position (seq_len - 1) is never directly in any prediction
    
    # Solution: Use the first sample's true values and work backwards
    # Or load from original CSV with proper scaling
    
    # Load scaler parameters from train data and apply to raw data
    from sklearn.preprocessing import StandardScaler
    
    df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
    
    # Excluded sensors
    exclude_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' not in exclude_sensors]
    
    # Fit scaler on train engines (1~70) with first 100 rows each
    train_data_list = []
    for eid in range(1, 71):
        engine_data = df[df['id'] == eid][sensor_cols].values[:100]
        train_data_list.append(engine_data)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_data_list, axis=0))
    
    # Get raw data for current engine and apply scaling
    engine_raw = df[df['id'] == engine_id][sensor_cols].values
    engine_scaled = scaler.transform(engine_raw)
    
    # Use the scaled raw data as ground truth
    reconstructed_gt = engine_scaled[:, channel]
    
    print(f"  Ground Truth reconstructed from CSV with scaling (length: {len(reconstructed_gt)})")

    # Figure setup
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Background: Full Ground Truth (gray)
    time_axis = np.arange(total_length)
    ax.plot(time_axis, reconstructed_gt, label='Ground Truth', color='gray', alpha=0.5, linewidth=1.5)
    
    # # 100행 기준선 (정상/비정상 구분)
    # ax.axvline(x=NORMAL_LIMIT, color='orange', linestyle='--', linewidth=2, 
    #            label=f'Normal Limit ({NORMAL_LIMIT})', alpha=0.8)
    
    # # 정상/비정상 영역 배경색
    # ax.axvspan(0, NORMAL_LIMIT, alpha=0.1, color='green', label='Normal Zone')
    # ax.axvspan(NORMAL_LIMIT, total_length, alpha=0.1, color='red', label='Degradation Zone')
    
    # 움직이는 선들
    line_lookback, = ax.plot([], [], label=f'Lookback ({seq_len})', 
                             color='blue', linewidth=2.5, marker='o', markersize=2)
    line_prediction, = ax.plot([], [], label=f'Prediction ({pred_len})', 
                               color='red', linewidth=2.5, marker='s', markersize=2)
    line_true_future, = ax.plot([], [], label='True', 
                                color='green', linewidth=2, linestyle='--')
    
    # 현재 위치 표시
    lookback_start_marker = ax.axvline(x=0, color='blue', linestyle=':', alpha=0.5)
    pred_end_marker = ax.axvline(x=0, color='red', linestyle=':', alpha=0.5)
    
    # 메트릭 텍스트 (Method 1 - 왼쪽 위)
    metrics_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                           verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.9),
                           family='monospace')
    
    # 메트릭 텍스트 (Method 2 - 왼쪽 아래)
    metrics_text2 = ax.text(0.02, 0.05, '', transform=ax.transAxes, fontsize=11,
                            verticalalignment='bottom', horizontalalignment='left',
                            bbox=dict(boxstyle='round,pad=0.5', fc='lightcyan', alpha=0.9),
                            family='monospace')
    
    # 그래프 설정
    ax.set_xlim(-5, total_length + 5)
    y_min, y_max = reconstructed_gt.min(), reconstructed_gt.max()
    y_margin = (y_max - y_min) * 0.15
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', fontsize=9, ncol=4)
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel(f'Sensor Value ({SENSOR_NAMES[channel]})', fontsize=12)
    
    title = ax.set_title(f'Engine {engine_id} | seq_len={seq_len} | pred_len={pred_len} | {SENSOR_NAMES[channel]}',
                         fontsize=14, fontweight='bold')
    
    # 방법 2: 각 시점별 겹치는 예측들의 오차 누적
    # overlap_mse_sum[t] = 시점 t에 대한 모든 예측의 squared error 합
    # overlap_mae_sum[t] = 시점 t에 대한 모든 예측의 absolute error 합
    # overlap_count[t] = 시점 t에 대한 예측 개수
    overlap_mse_sum = np.zeros(total_length)
    overlap_mae_sum = np.zeros(total_length)
    overlap_count = np.zeros(total_length)
    
    def update(frame):
        """프레임 업데이트 함수"""
        if frame >= num_samples:
            return line_lookback, line_prediction, line_true_future, metrics_text, metrics_text2
        
        # 현재 샘플의 prediction
        pred_values = engine_preds[frame, -pred_len:, channel]
        true_values = engine_trues[frame, -pred_len:, channel]
        
        # 위치 계산
        lookback_start = frame
        lookback_end = frame + seq_len
        pred_start = lookback_end
        pred_end = pred_start + pred_len
        
        # 1. Lookback Window
        lookback_x = np.arange(lookback_start, lookback_end)
        lookback_y = reconstructed_gt[lookback_start:lookback_end]
        line_lookback.set_data(lookback_x, lookback_y)
        
        # 2. Prediction
        pred_x = np.arange(pred_start, min(pred_end, total_length))
        line_prediction.set_data(pred_x, pred_values[:len(pred_x)])
        
        # 3. Ground Truth (예측 구간)
        line_true_future.set_data(pred_x, true_values[:len(pred_x)])
        
        # 마커 위치 업데이트
        lookback_start_marker.set_xdata([lookback_start])
        pred_end_marker.set_xdata([min(pred_end, total_length)])
        
        # 4. 메트릭 계산 (방법 1: pred_len 구간 평균)
        mse = np.mean((true_values - pred_values) ** 2)
        mae = np.mean(np.abs(true_values - pred_values))
        
        # 5. 방법 2: 각 시점별 겹치는 예측들의 오차 누적
        for j in range(pred_len):
            t = pred_start + j
            if t < total_length:
                error = pred_values[j] - true_values[j]
                overlap_mse_sum[t] += error ** 2
                overlap_mae_sum[t] += np.abs(error)
                overlap_count[t] += 1
        
        # 방법 2의 현재 시점(seq_len + frame) MSE/MAE 계산
        # Full overlap 시작: seq_len + pred_len - 1
        current_t = seq_len + frame  # 현재 lookback 끝 지점
        full_overlap_start = seq_len + pred_len - 1
        
        # seq_len 이후부터 동적으로 계산 (overlap count가 있으면 표시)
        if overlap_count[current_t] > 0:
            overlap_mse_curr = overlap_mse_sum[current_t] / overlap_count[current_t]
            overlap_mae_curr = overlap_mae_sum[current_t] / overlap_count[current_t]
            count_str = f"(count={int(overlap_count[current_t])})"
            overlap_str = f'Overlap MSE: {overlap_mse_curr:.5f} {count_str}\nOverlap MAE: {overlap_mae_curr:.5f}'
        else:
            overlap_str = f'Overlap MSE: (no prediction yet)\nOverlap MAE: -'
        
        # Method 1 텍스트 (왼쪽 위)
        metrics_text.set_text(
            f'--- Method 1 (pred_len avg) ---\n'
            f'Avg MSE: {mse:.5f}\n'
            f'Avg MAE: {mae:.5f}'
        )
        
        # Method 2 텍스트 (왼쪽 아래)
        metrics_text2.set_text(
            f'--- Method 2 (overlap avg) ---\n'
            f'{overlap_str}'
        )
        
        return line_lookback, line_prediction, line_true_future, metrics_text, metrics_text2
    
    # 애니메이션 생성
    print(f"애니메이션 생성 중... (총 {num_samples} 프레임)")
    ani = FuncAnimation(fig, update, frames=num_samples, blit=False, 
                        interval=FRAME_INTERVAL, repeat=True)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', dpi=DPI)
    print(f"애니메이션 저장 완료: {output_path}")
    
    plt.close()
    
    # 최종 메트릭 출력
    print(f"\n=== 엔진 {engine_id} 최종 결과 ===")
    
    # Method 1: 각 샘플의 pred_len 구간 평균 MSE/MAE의 전체 평균
    all_mse = []
    all_mae = []
    for frame in range(num_samples):
        pred_values = engine_preds[frame, -pred_len:, channel]
        true_values = engine_trues[frame, -pred_len:, channel]
        all_mse.append(np.mean((true_values - pred_values) ** 2))
        all_mae.append(np.mean(np.abs(true_values - pred_values)))
    
    print(f"--- Method 1 (pred_len 구간 평균) ---")
    print(f"  전체 MSE: {np.mean(all_mse):.5f}")
    print(f"  전체 MAE: {np.mean(all_mae):.5f}")
    
    # 방법 2: 겹치는 예측이 있는 구간의 평균 계산
    full_overlap_start = seq_len + pred_len - 1
    valid_mask = overlap_count > 0
    
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        overlap_mse_values = overlap_mse_sum[valid_indices] / overlap_count[valid_indices]
        overlap_mae_values = overlap_mae_sum[valid_indices] / overlap_count[valid_indices]
        
        # Full overlap 구간 (pred_len개의 예측이 겹치는 구간)
        full_overlap_mask = overlap_count >= pred_len
        if np.any(full_overlap_mask):
            full_indices = np.where(full_overlap_mask)[0]
            full_mse = np.mean(overlap_mse_sum[full_indices] / overlap_count[full_indices])
            full_mae = np.mean(overlap_mae_sum[full_indices] / overlap_count[full_indices])
            print(f"--- Method 2 (Full Overlap, count={pred_len}) ---")
            print(f"  전체 MSE: {full_mse:.5f}")
            print(f"  전체 MAE: {full_mae:.5f}")
        else:
            print(f"--- Method 2 (No Full Overlap, max_count={int(overlap_count.max())}) ---")
            print(f"  전체 MSE: {np.mean(overlap_mse_values):.5f}")
            print(f"  전체 MAE: {np.mean(overlap_mae_values):.5f}")


def main():
    print("=" * 60)
    print("C-MAPSS Sliding Window Prediction 시각화")
    print("=" * 60)
    print(f"설정:")
    print(f"  - seq_len: {SEQ_LEN}")
    print(f"  - pred_len: {PRED_LEN}")
    print(f"  - Channel: {SENSOR_NAMES[CHANNEL]} (index {CHANNEL})")
    print(f"  - Engine Index: {ENGINE_INDEX}")
    print(f"  - Normal Pattern Standard: {NORMAL_LIMIT}rows")
    print("=" * 60)
    
    # 결과 경로 찾기 (seq_len과 pred_len으로 구분)
    # 기존
    # result_path = find_result_path(SEQ_LEN, PRED_LEN)
    
    # 절대 경로
    folder_name = "100_train_FD001_seq48_pred48_iTransformer_CMAPSS_Normal_M_ft48_sl24_ll48_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"
    result_path = os.path.join('./results/', folder_name)

    if result_path is None:
        return
    
    print(f"\nResult path: {result_path}")
    
    # 데이터 로드
    trues, preds = load_data(result_path)
    if trues is None:
        return
    
    # 엔진 정보 로드
    engine_info, test_engine_ids = get_engine_info()
    if engine_info is None:
        return
    
    print(f"\n테스트 엔진 정보:")
    for eid in test_engine_ids:
        print(f"  - 엔진 {eid}: {engine_info[eid]['length']}행")
    
    # 출력 파일명
    output_filename = f"100_engine{test_engine_ids[ENGINE_INDEX]}_seq{SEQ_LEN}_pred{PRED_LEN}_{SENSOR_NAMES[CHANNEL]}.gif"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # 애니메이션 생성
    create_animation_for_engine(
        trues, preds, ENGINE_INDEX, CHANNEL, SEQ_LEN, PRED_LEN,
        engine_info, test_engine_ids, output_path
    )


if __name__ == "__main__":
    main()
