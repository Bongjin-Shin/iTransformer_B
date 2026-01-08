import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

"""
Dataset_CMAPSS 시각화 (seq_len = 12, 24, 48)
- Ground Truth 전체 데이터가 배경으로 표시
- Lookback window (입력)가 파란색으로 표시
- Prediction (예측)이 빨간색으로 표시
- MSE, MAE 메트릭 표시
"""

# ============================================================
# 설정
# ============================================================
# 시각화할 seq_len 선택 (12, 24, 48 중 하나)
SEQ_LEN = 48  # 변경 가능

# seq_len에 따른 설정
SEQ_LEN_CONFIG = {
    12: {'label_len': 6, 'pred_len': 24, 'model_id': 'train_FD001_seq12'},
    24: {'label_len': 12, 'pred_len': 24, 'model_id': 'train_FD001_seq24'},
    48: {'label_len': 24, 'pred_len': 24, 'model_id': 'train_FD001_seq48'},
}

config = SEQ_LEN_CONFIG[SEQ_LEN]
LABEL_LEN = config['label_len']
PRED_LEN = config['pred_len']
MODEL_ID = config['model_id']

# 센서 이름 (제외 센서: s1, s5, s6, s10, s16, s18, s19)
SENSOR_NAMES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']

# 시각화할 채널 (0~13)
CHANNEL = 0

# 애니메이션 설정
START_FRAME = 0
NUM_FRAMES = 50  # 렌더링할 프레임 수

# 출력 폴더
OUTPUT_DIR = './gif/cmapss_standard/'


def find_result_path(model_id):
    """결과 폴더 경로 찾기"""
    results_dir = './results/'
    
    if not os.path.exists(results_dir):
        print(f"오류: results 폴더가 없습니다: {results_dir}")
        return None
    
    # model_id를 포함하는 폴더 찾기
    for folder in os.listdir(results_dir):
        if model_id in folder and 'CMAPSS' in folder:
            return os.path.join(results_dir, folder)
    
    print(f"오류: {model_id}를 포함하는 결과 폴더를 찾을 수 없습니다.")
    print(f"사용 가능한 폴더:")
    for folder in os.listdir(results_dir):
        if 'CMAPSS' in folder:
            print(f"  - {folder}")
    return None


def load_data(base_path):
    """데이터 로드"""
    try:
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        print(f"데이터 로드 성공: {base_path}")
        print(f"  trues shape: {trues.shape}")
        print(f"  preds shape: {preds.shape}")
        return trues, preds
    except FileNotFoundError as e:
        print(f"오류: 파일을 찾을 수 없습니다: {e}")
        return None, None


def reconstruct_signal(trues, channel):
    """Ground Truth 전체 시그널 재구성"""
    num_indices = trues.shape[0]
    
    if num_indices > 1:
        # 첫 번째 윈도우 전체 + 나머지 윈도우의 마지막 포인트
        first_window = trues[0, :, channel]
        last_points = trues[1:, -1, channel]
        reconstructed = np.concatenate((first_window, last_points))
    else:
        reconstructed = trues[0, :, channel]
    
    return reconstructed


def create_animation(trues, preds, channel, seq_len, pred_len, output_path):
    """애니메이션 생성"""
    num_indices, window_size, num_channels = trues.shape
    
    # 전체 시그널 재구성
    reconstructed_signal = reconstruct_signal(trues, channel)
    total_length = len(reconstructed_signal)
    
    print(f"재구성된 신호 길이: {total_length}")
    
    # Figure 설정
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 배경: 전체 Ground Truth (회색)
    ax.plot(reconstructed_signal, label='Ground Truth (전체)', color='gray', alpha=0.5, linewidth=1)
    
    # 움직이는 선들
    line_lookback, = ax.plot([], [], label=f'Lookback Window ({seq_len})', color='blue', linewidth=2)
    line_prediction, = ax.plot([], [], label=f'Prediction ({pred_len})', color='red', linewidth=2)
    line_true_future, = ax.plot([], [], label='Ground Truth (예측 구간)', color='green', linewidth=2, linestyle='--')
    
    # 메트릭 텍스트
    metrics_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, fontsize=12,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.8))
    
    # 그래프 설정
    ax.set_xlim(0, total_length)
    y_min, y_max = reconstructed_signal.min(), reconstructed_signal.max()
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time Step (Cycle)', fontsize=12)
    ax.set_ylabel(f'Sensor Value ({SENSOR_NAMES[channel]})', fontsize=12)
    title = ax.set_title('')
    
    def update(frame_index):
        """프레임 업데이트 함수"""
        # 현재 예측 인덱스
        pred_idx = START_FRAME + frame_index
        
        if pred_idx >= num_indices:
            return line_lookback, line_prediction, line_true_future, title, metrics_text
        
        # 예측 시작 위치 계산
        # 각 샘플의 예측은 pred_idx 위치에서 시작
        pred_start_pos = pred_idx + pred_len
        
        # 1. Lookback Window (입력 구간)
        lookback_start = pred_start_pos - pred_len - seq_len
        lookback_end = pred_start_pos - pred_len
        
        if lookback_start < 0:
            lookback_start = 0
        
        if lookback_end <= total_length:
            lookback_x = np.arange(lookback_start, lookback_end)
            lookback_y = reconstructed_signal[lookback_start:lookback_end]
            line_lookback.set_data(lookback_x, lookback_y)
        
        # 2. Prediction (예측값)
        pred_end_pos = min(pred_start_pos + pred_len, total_length)
        pred_x = np.arange(pred_start_pos, pred_end_pos)
        pred_y = preds[pred_idx, :len(pred_x), channel]
        line_prediction.set_data(pred_x, pred_y)
        
        # 3. Ground Truth (예측 구간의 실제값)
        true_y = trues[pred_idx, :len(pred_x), channel]
        line_true_future.set_data(pred_x, true_y)
        
        # 4. 메트릭 계산
        mse = np.mean((true_y - pred_y) ** 2)
        mae = np.mean(np.abs(true_y - pred_y))
        
        title.set_text(f'C-MAPSS | seq_len={seq_len} | Channel: {SENSOR_NAMES[channel]} | Frame: {pred_idx}')
        metrics_text.set_text(f'MSE: {mse:.5f}\nMAE: {mae:.5f}')
        
        return line_lookback, line_prediction, line_true_future, title, metrics_text
    
    # 프레임 범위 확인
    end_frame = START_FRAME + NUM_FRAMES
    if end_frame > num_indices:
        print(f"경고: 요청된 프레임 범위({end_frame})가 데이터 크기({num_indices})를 초과합니다.")
        end_frame = num_indices
        actual_frames = end_frame - START_FRAME
    else:
        actual_frames = NUM_FRAMES
    
    animation_frames = np.arange(actual_frames)
    ani = FuncAnimation(fig, update, frames=animation_frames, blit=True, interval=200)
    
    # 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ani.save(output_path, writer='pillow', dpi=100)
    print(f"애니메이션 저장 완료: {output_path}")
    
    plt.close()


def create_static_plot(trues, preds, channel, seq_len, pred_len, sample_idx, output_path):
    """정적 플롯 생성 (특정 샘플에 대해)"""
    num_indices, window_size, num_channels = trues.shape
    
    # 전체 시그널 재구성
    reconstructed_signal = reconstruct_signal(trues, channel)
    total_length = len(reconstructed_signal)
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # 배경: 전체 Ground Truth
    ax.plot(reconstructed_signal, label='Ground Truth (전체)', color='gray', alpha=0.5, linewidth=1)
    
    # 예측 시작 위치
    pred_start_pos = sample_idx + pred_len
    
    # Lookback Window
    lookback_start = max(0, pred_start_pos - pred_len - seq_len)
    lookback_end = pred_start_pos - pred_len
    
    if lookback_end <= total_length:
        lookback_x = np.arange(lookback_start, lookback_end)
        lookback_y = reconstructed_signal[lookback_start:lookback_end]
        ax.plot(lookback_x, lookback_y, label=f'Lookback ({seq_len})', color='blue', linewidth=2)
    
    # Prediction
    pred_end_pos = min(pred_start_pos + pred_len, total_length)
    pred_x = np.arange(pred_start_pos, pred_end_pos)
    pred_y = preds[sample_idx, :len(pred_x), channel]
    ax.plot(pred_x, pred_y, label=f'Prediction ({pred_len})', color='red', linewidth=2)
    
    # Ground Truth (예측 구간)
    true_y = trues[sample_idx, :len(pred_x), channel]
    ax.plot(pred_x, true_y, label='Ground Truth (예측 구간)', color='green', linewidth=2, linestyle='--')
    
    # 메트릭
    mse = np.mean((true_y - pred_y) ** 2)
    mae = np.mean(np.abs(true_y - pred_y))
    
    ax.set_xlim(0, total_length)
    y_min, y_max = reconstructed_signal.min(), reconstructed_signal.max()
    y_margin = (y_max - y_min) * 0.1
    ax.set_ylim(y_min - y_margin, y_max + y_margin)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time Step (Cycle)', fontsize=12)
    ax.set_ylabel(f'Sensor Value ({SENSOR_NAMES[channel]})', fontsize=12)
    ax.set_title(f'C-MAPSS | seq_len={seq_len} | Channel: {SENSOR_NAMES[channel]} | Sample: {sample_idx}\nMSE: {mse:.5f} | MAE: {mae:.5f}')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"정적 플롯 저장 완료: {output_path}")
    plt.close()


def visualize_all_seq_lens(channel=0):
    """모든 seq_len에 대해 시각화"""
    for seq_len, config in SEQ_LEN_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len} 처리 중...")
        print(f"{'='*60}")
        
        model_id = config['model_id']
        pred_len = config['pred_len']
        
        # 결과 폴더 찾기
        base_path = find_result_path(model_id)
        if base_path is None:
            print(f"seq_len={seq_len} 건너뜀 (결과 폴더 없음)")
            continue
        
        # 데이터 로드
        trues, preds = load_data(base_path)
        if trues is None:
            continue
        
        # 애니메이션 생성
        output_gif = os.path.join(OUTPUT_DIR, f'cmapss_seq{seq_len}_ch{channel}_{SENSOR_NAMES[channel]}.gif')
        create_animation(trues, preds, channel, seq_len, pred_len, output_gif)
        
        # 정적 플롯 (샘플 몇 개)
        for sample_idx in [0, 10, 20]:
            if sample_idx < trues.shape[0]:
                output_png = os.path.join(OUTPUT_DIR, f'cmapss_seq{seq_len}_ch{channel}_sample{sample_idx}.png')
                create_static_plot(trues, preds, channel, seq_len, pred_len, sample_idx, output_png)


def main():
    """메인 함수"""
    print("C-MAPSS Standard 시각화")
    print(f"설정: SEQ_LEN={SEQ_LEN}, CHANNEL={CHANNEL} ({SENSOR_NAMES[CHANNEL]})")
    
    # 단일 seq_len 시각화
    base_path = find_result_path(MODEL_ID)
    if base_path is None:
        print("\n모든 seq_len에 대해 시각화를 시도합니다...")
        visualize_all_seq_lens(CHANNEL)
        return
    
    trues, preds = load_data(base_path)
    if trues is None:
        return
    
    # 애니메이션 생성
    output_gif = os.path.join(OUTPUT_DIR, f'cmapss_seq{SEQ_LEN}_ch{CHANNEL}_{SENSOR_NAMES[CHANNEL]}.gif')
    create_animation(trues, preds, CHANNEL, SEQ_LEN, PRED_LEN, output_gif)
    
    # 정적 플롯
    for sample_idx in [0, 10, 20]:
        if sample_idx < trues.shape[0]:
            output_png = os.path.join(OUTPUT_DIR, f'cmapss_seq{SEQ_LEN}_ch{CHANNEL}_sample{sample_idx}.png')
            create_static_plot(trues, preds, CHANNEL, SEQ_LEN, PRED_LEN, sample_idx, output_png)


if __name__ == '__main__':
    main()
