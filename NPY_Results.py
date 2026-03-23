import numpy as np
import matplotlib.pyplot as plt

# --- 1. 데이터 로딩 및 준비 ---
data_type = 'Weather'  # 'ETT' 또는 'Weather' 중 선택
base_path = './results/A-3_seq96_pred96_iTransformer_NPY_S_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/'

# trues.npy만 사용
trues = np.load(base_path + 'true.npy')

# --- 설정 값을 명확하게 정의 ---
num_indices, window_size, num_channels = trues.shape
print(f"인덱스 수: {num_indices}, 윈도우 크기: {window_size}, 채널 수: {num_channels}")

input_length = 96
prediction_length = window_size

# 분석할 채널을 선택 (예: 마지막 채널)
channel = 0

# '실제 미래값(trues)'들을 이어붙여 전체 타임라인의 실제값(Ground Truth)을 재구성
if num_indices > 1:
    first_window = trues[0, :, channel]
    last_points = trues[1:, -1, channel]
    reconstructed_signal = np.concatenate((first_window, last_points))
else:
    reconstructed_signal = trues[0, :, channel]

# --- 2. 그래프 그리기 ---
plt.figure(figsize=(12, 4))
plt.plot(reconstructed_signal, label=f"True Channel {channel}")
plt.title(f"Ground Truth Signal (Channel {channel})")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()