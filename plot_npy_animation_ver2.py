# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # --- 1. 데이터 로딩 및 준비 ---
# try:
#     # 사용자 환경에 맞는 실제 파일 경로로 수정해주세요.
#     data_type = 'Weather'  # 예시: 'Appliances', 'Exchange', 'ECL'
#     base_path = './results/weather_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/'
    
#     # trues.npy는 '실제 미래값(Ground Truth)', preds.npy는 '예측 미래값(Prediction)' 입니다.
#     trues = np.load(base_path + 'true.npy')
#     preds = np.load(base_path + 'pred.npy')

#     # --- 설정 값을 명확하게 정의 ---
#     num_indices, window_size, num_channels = trues.shape
#     print(f"인덱스 수: {num_indices}, 윈도우 크기: {window_size}, 채널 수: {num_channels}")

#     input_length = 96      # 모델의 입력 길이
#     prediction_length = window_size # 모델의 예측 길이 (trues와 preds의 길이)
    
#     # 분석할 채널을 선택합니다.
#     channel = -1

#     # '실제 미래값(trues)'들을 이어붙여 전체 타임라인의 실제값(Ground Truth)을 재구성합니다.
#     if num_indices > 1:
#         first_window = trues[0, :, channel]
#         last_points = trues[1:, -1, channel]
#         reconstructed_signal = np.concatenate((first_window, last_points))
#     else:
#         reconstructed_signal = trues[0, :, channel]

#     # --- 2. 애니메이션 설정 ---
#     fig, ax = plt.subplots(figsize=(50, 12))

#     # 배경: 재구성된 전체 실제 데이터를 회색선으로 그립니다.
#     ax.plot(reconstructed_signal, label='Ground Truth', color='gray', alpha=0.7)

#     # 움직이는 선들:
#     # '가상'의 과거 입력(파란색), 예측 미래값(빨간색)
#     line_past_input, = ax.plot([], [], label=f'Input : {input_length}', color='blue', linewidth=0.7)
#     line_pred_future, = ax.plot([], [], label=f'Predict : {prediction_length}', color='red', linewidth=0.7)

#     metrics_text = ax.text(0.03, 0.9, '', transform=ax.transAxes, fontsize=12,
#                        verticalalignment='top', horizontalalignment='right',
#                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

#     # 그래프 축 범위 및 레이블 고정
#     ax.set_xlim(0, len(reconstructed_signal))
#     ax.set_ylim(reconstructed_signal.min() * 0.95, reconstructed_signal.max() * 1.05)
#     ax.grid(True)
#     ax.legend(loc='upper left')
#     ax.set_xlabel('Time Step', fontsize=12)
#     ax.set_ylabel('Value', fontsize=12)
#     title = ax.set_title('')

#     # --- 3. 애니메이션 업데이트 함수 정의 ---
#     def update(frame_index):
#         # 각 예측 창의 시작점을 계산합니다. 
#         prediction_start_pos = frame_index + input_length  # 96부터 시작

#         # MSE와 MAE 계산을 위한 실제값과 예측값을 가져옵니다.
#         true_vals = trues[prediction_start_pos, :, channel]
#         pred_vals = preds[prediction_start_pos, :, channel]

#         # 1. '가상'의 과거 입력 데이터 그리기 (reconstructed_signal에서 가져오기)
#         past_start_pos = prediction_start_pos - input_length
#         past_end_pos = prediction_start_pos
        
#         # reconstructed_signal에서 음수 인덱싱을 방지하기 위한 처리
#         if past_start_pos < 0:
#             past_start_pos = 0

#         past_x = np.arange(past_start_pos, past_end_pos)
#         past_y = reconstructed_signal[past_start_pos:past_end_pos]
#         line_past_input.set_data(past_x, past_y)

#         # 2. 실제 미래값과 예측 미래값 그리기
#         future_x = np.arange(prediction_start_pos, prediction_start_pos + prediction_length)  # 96 ~ 192

#         # 예측 미래값 (preds 배열에서 직접 가져옴)
#         future_y = preds[prediction_start_pos, :, channel]
#         line_pred_future.set_data(future_x, future_y)
        
#         title.set_text(f'{data_type} Time Step: {frame_index}')

#         mse = np.mean((true_vals - pred_vals)**2)
#         mae = np.mean(np.abs(true_vals - pred_vals))
#         metrics_text.set_text(f'MSE: {mse:.5f}\nMAE: {mae:.5f}')

#         return line_past_input, line_pred_future, title, metrics_text

#     # --- 4. 애니메이션 생성 및 저장 ---
#     # `frames`는 trues, preds 배열의 인덱스 수(num_indices)를 넘지 않도록 설정하는 것이 안전합니다.
#     ani = FuncAnimation(fig, update, frames=800, blit=True, interval=100)
#     output_filename = f'{data_type}_OT_prediction_animation_{input_length}_{prediction_length}.gif'
#     ani.save(output_filename, writer='pillow', dpi=100)
    
#     print(f"'{output_filename}' 파일 생성이 완료되었습니다.")

# except FileNotFoundError:
#     print(f"오류: 지정된 경로에 파일이 없습니다. 파일 경로를 다시 확인해주세요: {base_path}")
# except Exception as e:
#     print(f"오류가 발생했습니다: {e}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys # <--- 변경/추가된 부분: 시스템 종료를 위해 추가

# --- 1. 데이터 로딩 및 준비 ---
try:
    # 사용자 환경에 맞는 실제 파일 경로로 수정해주세요.
    data_type = 'Weather'  # 예시: 'Appliances', 'Exchange', 'ECL'
    base_path = './results/weather_96_720_iTransformer_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/'
    
    # trues.npy는 '실제 미래값(Ground Truth)', preds.npy는 '예측 미래값(Prediction)' 입니다.
    trues = np.load(base_path + 'true.npy')
    preds = np.load(base_path + 'pred.npy')

    # --- 설정 값을 명확하게 정의 ---
    num_indices, window_size, num_channels = trues.shape
    print(f"인덱스 수: {num_indices}, 윈도우 크기: {window_size}, 채널 수: {num_channels}")

    input_length = 96      # 모델의 입력 길이
    prediction_length = window_size # 모델의 예측 길이 (trues와 preds의 길이)
    
    # 분석할 채널을 선택합니다.
    channel = 14

    # '실제 미래값(trues)'들을 이어붙여 전체 타임라인의 실제값(Ground Truth)을 재구성합니다.
    if num_indices > 1:
        first_window = trues[0, :, channel]
        last_points = trues[1:, -1, channel]
        reconstructed_signal = np.concatenate((first_window, last_points))
    else:
        reconstructed_signal = trues[0, :, channel]

    # --- 2. 애니메이션 설정 ---
    fig, ax = plt.subplots(figsize=(50, 12))

    # 배경: 재구성된 전체 실제 데이터를 회색선으로 그립니다.
    ax.plot(reconstructed_signal, label='Ground Truth', color='gray', alpha=0.7)

    # 움직이는 선들:
    # '가상'의 과거 입력(파란색), 예측 미래값(빨간색)
    line_past_input, = ax.plot([], [], label=f'Input : {input_length}', color='blue', linewidth=0.7)
    line_pred_future, = ax.plot([], [], label=f'Predict : {prediction_length}', color='red', linewidth=0.7)

    metrics_text = ax.text(0.03, 0.9, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # 그래프 축 범위 및 레이블 고정
    ax.set_xlim(0, len(reconstructed_signal))
    ax.set_ylim(reconstructed_signal.min(), reconstructed_signal.max())
    ax.grid(True)
    ax.legend(loc='upper left')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    title = ax.set_title('')

    # --- 3. 애니메이션 업데이트 함수 정의 ---
    def update(frame_index):
        # 각 예측 창의 시작점을 계산합니다.
        # 원본 코드의 논리를 그대로 따릅니다.
        prediction_start_pos = frame_index + input_length + 1  # 0 + 96 + 1 = 97부터 시작

        # 원본 코드에서는 prediction_start_pos를 인덱스로 사용했으나,
        # trues와 preds 배열의 인덱스는 0부터 시작하는 frame_index를 사용하는 것이 일반적입니다.
        # 혼란을 막기 위해 frame_index를 직접 사용하도록 수정합니다.
        true_vals = trues[prediction_start_pos, :, channel]
        pred_vals = preds[prediction_start_pos, :, channel]

        # 1. '가상'의 과거 입력 데이터 그리기 (reconstructed_signal에서 가져오기)
        past_start_pos = prediction_start_pos - input_length - 1  # 0부터 시작
        past_end_pos = prediction_start_pos  # 96까지
        
        # reconstructed_signal에서 음수 인덱싱을 방지하기 위한 처리
        if past_start_pos < 0:
            past_start_pos = 0

        past_x = np.arange(past_start_pos, past_end_pos)
        past_y = reconstructed_signal[past_start_pos:past_end_pos]
        line_past_input.set_data(past_x, past_y)
    
        # 2. 예측 미래값 그리기
        future_x = np.arange(prediction_start_pos, prediction_start_pos + prediction_length)
        future_y = preds[prediction_start_pos, :, channel] # frame_index를 사용하여 현재 프레임의 예측값을 가져옵니다.
        line_pred_future.set_data(future_x, future_y)
        
        title.set_text(f'{data_type} Time Step: {frame_index}')

        mse = np.mean((true_vals - pred_vals)**2)
        mae = np.mean(np.abs(true_vals - pred_vals))
        metrics_text.set_text(f'MSE: {mse:.5f}\nMAE: {mae:.5f}')

        return line_past_input, line_pred_future, title, metrics_text

    # --- 4. 애니메이션 생성 및 저장 ---
    
    # --- 애니메이션 시작 및 종료 프레임 설정 --- # <--- 변경/추가된 부분
    start_frame = 5300
    num_frames_to_render = 300  # 원본 코드와 동일하게 800개 프레임을 렌더링
    end_frame = start_frame + num_frames_to_render

    # --- 요청된 프레임 범위가 유효한지 확인 --- # <--- 변경/추가된 부분
    # update 함수가 접근할 최대 인덱스는 end_frame - 1 입니다.
    # 이 값이 trues, preds 배열의 크기(num_indices)보다 작아야 합니다.
    if end_frame > num_indices:
        print(f"\n오류: 요청된 프레임 범위가 데이터 크기를 초과합니다.")
        print(f"데이터의 최대 인덱스 (num_indices): {num_indices}")
        print(f"요청된 마지막 프레임 인덱스: {end_frame - 1}")
        print("start_frame 또는 num_frames_to_render 값을 줄여주세요.")
        sys.exit() # 프로그램 종료

    # `frames` 인자에 시작점과 종료점을 지정한 배열을 전달합니다.
    animation_frames = np.arange(start_frame, end_frame)
    ani = FuncAnimation(fig, update, frames=animation_frames, blit=True, interval=100)
    
    # 파일 이름에 시작 프레임 정보를 추가하여 구별하기 쉽게 합니다.
    # output_filename = f'{data_type}_OT_prediction_animation_{input_length}_{prediction_length}_start{start_frame}.gif'
    output_filename = f'channel_{channel}.gif'  # 채널 정보 추가
    ani.save(output_filename, writer='pillow', dpi=100)
    
    print(f"\n'{output_filename}' 파일 생성이 완료되었습니다.")
    print(f"애니메이션은 {start_frame} 프레임부터 {end_frame - 1} 프레임까지 생성되었습니다.")

except FileNotFoundError:
    print(f"오류: 지정된 경로에 파일이 없습니다. 파일 경로를 다시 확인해주세요: {base_path}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")