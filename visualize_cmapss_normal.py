"""애니메이션 만드는 코드"""


"""
C-MAPSS Normal Region (Limit 100) Prediction Visualization
시각화 대상: Dataset_CMAPSS_NormalOnly로 학습/테스트된 결과 (모든 엔진 100행 제한)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
from sklearn.preprocessing import StandardScaler

# ============================================================
# 1. Configuration (사용자 설정)
# ============================================================

# [중요] 시각화할 결과 폴더 이름 (results 폴더 내의 이름)
# 예: 100_train_FD001_seq48_pred48_...
FOLDER_NAME = "100_train_FD001_seq48_pred24_iTransformer_CMAPSS_M_ft48_sl12_ll24_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0"

# 학습 설정과 동일하게 맞춰주세요
SEQ_LEN = 48       # Input sequence length
PRED_LEN = 24      # Prediction sequence length
TRAIN_LIMIT = 100  # Dataset에서 설정한 limit (보통 100)

# 시각화할 센서 및 엔진 설정
SENSOR_NAMES = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
CHANNEL = 13      # 보고 싶은 센서 인덱스 (0~13), 5는 's9'
ENGINE_IDX_IN_TEST = 10  # Test set 내의 엔진 순서 (0: Engine 81, 1: Engine 82, ...)

# 애니메이션 설정
FRAME_INTERVAL = 200  # ms
OUTPUT_DIR = './gif/cmapss_normal/'

# ============================================================
# 2. Helper Functions
# ============================================================

def load_data(folder_name):
    """결과 폴더에서 numpy 데이터 로드"""
    base_path = os.path.join('./results/', folder_name)
    try:
        trues = np.load(os.path.join(base_path, 'true.npy'))
        preds = np.load(os.path.join(base_path, 'pred.npy'))
        print(f"✅ Data loaded: {base_path}")
        print(f"   Shape - True: {trues.shape}, Pred: {preds.shape}")
        return trues, preds
    except FileNotFoundError:
        print(f"❌ Error: Files not found in {base_path}")
        return None, None

def get_scaler_and_raw_data(engine_id, sensor_names, train_limit=100):
    """
    Dataset 클래스와 동일한 방식으로 Scaler를 학습시키고
    선택한 엔진의 Raw Data(0~100)를 스케일링하여 반환
    """
    df = pd.read_csv('./dataset/CMAPSSData/train_FD001_with_columns.csv')
    
    # Exclude sensors logic (Dataset 클래스와 동일)
    exclude_sensors = ['s1', 's5', 's6', 's10', 's16', 's18', 's19']
    sensor_cols = [f's{i}' for i in range(1, 22) if f's{i}' not in exclude_sensors]
    
    # 1. Scaler Fit (Train engines 1~70, limit 100)
    train_data_list = []
    for eid in range(1, 71):
        df_e = df[df['id'] == eid].iloc[:train_limit]
        train_data_list.append(df_e[sensor_cols].values)
    
    scaler = StandardScaler()
    scaler.fit(np.concatenate(train_data_list, axis=0))
    
    # 2. Target Engine Data Transform
    df_target = df[df['id'] == engine_id].reset_index(drop=True)
    # 100행까지만 자름
    target_len = min(train_limit, len(df_target))
    raw_values = df_target[sensor_cols].values[:target_len]
    scaled_values = scaler.transform(raw_values)
    
    return scaled_values, sensor_cols

def calculate_samples_per_engine(seq_len, pred_len, limit=100):
    """
    Dataset 로직에 따른 엔진 당 샘플 수 계산
    Logic: max_idx = len(data) - seq_len - pred_len
           samples = max_idx - min_idx + 1
           (min_idx is usually 0 if label_len <= seq_len)
    """
    # Dataset 클래스는 모든 엔진을 limit(100)으로 잘랐음.
    # 따라서 모든 테스트 엔진의 길이는 100이라고 가정 (FD001의 최소 길이는 >100이므로)
    data_len = limit
    
    # Dataset __read_data__ 로직 참조
    # max_idx = len(data) - self.seq_len - self.pred_len
    # min_idx = max(0, self.label_len - self.seq_len) -> 보통 0
    
    max_idx = data_len - seq_len - pred_len
    
    # 샘플 개수 (start_idx 기준 0 ~ max_idx)
    num_samples = max_idx + 1
    
    return max(0, num_samples)

def main():
    print("=" * 60)
    print(" C-MAPSS Normal (Limit 100) Visualization")
    print("=" * 60)
    
    # 1. Load Result Data
    trues_all, preds_all = load_data(FOLDER_NAME)
    if trues_all is None: return

    # 2. Calculate Samples per Engine
    samples_per_engine = calculate_samples_per_engine(SEQ_LEN, PRED_LEN, TRAIN_LIMIT)
    print(f"⚙️ Config: Seq={SEQ_LEN}, Pred={PRED_LEN}, Limit={TRAIN_LIMIT}")
    print(f"📊 Calculated samples per engine: {samples_per_engine}")
    
    if samples_per_engine <= 0:
        print("❌ Error: seq_len + pred_len > train_limit (100). No samples can be generated.")
        return

    # 3. Select Engine Data
    # Test engines: 81 ~ 100
    test_engine_ids = list(range(81, 101))
    target_engine_id = test_engine_ids[ENGINE_IDX_IN_TEST]
    
    # Index slicing for the specific engine
    start_idx = ENGINE_IDX_IN_TEST * samples_per_engine
    end_idx = start_idx + samples_per_engine
    
    if end_idx > len(preds_all):
        print(f"❌ Error: Calculated index range ({start_idx}~{end_idx}) exceeds data length ({len(preds_all)}).")
        print("   Check if the loaded result matches the SEQ/PRED/LIMIT config.")
        return

    # Extract predictions for this engine
    # shape: (samples, pred_len, channels)
    eng_preds = preds_all[start_idx:end_idx]
    
    # 4. Get Ground Truth (Background)
    # 100행짜리 전체 Ground Truth를 CSV에서 다시 만들어옵니다 (Scaling 적용)
    gt_scaled, loaded_sensors = get_scaler_and_raw_data(target_engine_id, SENSOR_NAMES, TRAIN_LIMIT)
    gt_series = gt_scaled[:, CHANNEL]
    
    print(f"\nVisualizing Engine {target_engine_id} ({SENSOR_NAMES[CHANNEL]})")
    print(f" - GT Length: {len(gt_series)}")
    print(f" - Num Samples (Windows): {len(eng_preds)}")

    # ============================================================
    # 3. Animation
    # ============================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # X축: 0 ~ 100
    x_axis = np.arange(len(gt_series))
    
    # Background: Ground Truth
    ax.plot(x_axis, gt_series, color='gray', alpha=0.4, linewidth=2, label='Ground Truth (Normal 100)')
    
    # Dynamic Elements
    line_lookback, = ax.plot([], [], color='blue', marker='o', markersize=3, label='Lookback (Input)')
    line_pred, = ax.plot([], [], color='red', marker='x', markersize=3, label='Prediction')
    
    # Metrics Text
    text_metrics = ax.text(0.02, 0.95, '', transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    def init():
        ax.set_xlim(0, TRAIN_LIMIT)
        # Y축 범위 여유있게
        y_min, y_max = gt_series.min(), gt_series.max()
        margin = (y_max - y_min) * 0.2
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_title(f"Engine {target_engine_id} | {SENSOR_NAMES[CHANNEL]} | Normal Only (0~100)", fontsize=14)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Standardized Sensor Value")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        return line_lookback, line_pred, text_metrics

    def update(frame):
        # frame is the sample index (0 ~ samples_per_engine-1)
        
        # Calculate time steps
        # Dataset logic: 
        #   Sample 0: Lookback [0 : seq], Pred [seq : seq+pred]
        #   Sample 1: Lookback [1 : seq+1], Pred [seq+1 : seq+pred+1]
        #   ...
        start_t = frame
        mid_t = start_t + SEQ_LEN
        end_t = mid_t + PRED_LEN
        
        # 1. Lookback (Input)
        # GT 데이터에서 가져옴 (모델 입력이 실제 데이터였으므로)
        lb_x = np.arange(start_t, mid_t)
        lb_y = gt_series[start_t : mid_t]
        line_lookback.set_data(lb_x, lb_y)
        
        # 2. Prediction
        # 모델 출력 결과 사용
        pred_x = np.arange(mid_t, end_t)
        pred_y = eng_preds[frame, :, CHANNEL]
        line_pred.set_data(pred_x, pred_y)
        
        # 3. Metrics (MSE calculation for this window)
        # 실제 미래 값 (Ground Truth)
        true_y = gt_series[mid_t : end_t]
        # 데이터 끝부분 처리 (만약 인덱스가 범위를 벗어나면)
        if len(true_y) == len(pred_y):
            mse = np.mean((true_y - pred_y) ** 2)
            mae = np.mean(np.abs(true_y - pred_y))
            text_metrics.set_text(f"Step: {frame}\nRange: {mid_t}~{end_t}\nMSE: {mse:.4f}\nMAE: {mae:.4f}")
        
        return line_lookback, line_pred, text_metrics

    ani = FuncAnimation(fig, update, frames=len(eng_preds), init_func=init, 
                        blit=True, interval=FRAME_INTERVAL, repeat=True)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, f"Normal_Eng{target_engine_id}_{SENSOR_NAMES[CHANNEL]}_seq{SEQ_LEN}_pred{PRED_LEN}.gif")
    ani.save(save_path, writer='pillow', fps=5)
    print(f"💾 Animation Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()