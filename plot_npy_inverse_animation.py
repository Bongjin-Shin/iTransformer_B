######################################################### M 데이터 inverse 하기 ##########################################
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import os

# # === 설정 ===
# root_path = './dataset/exchange_rate/'
# data_path = 'exchange_rate.csv'
# target = 'OT'   # 타겟 변수 이름
# features = 'M'      # 'S', 'M', 'MS' 중 선택

# # === 1. pred.npy 파일 불러오기 ===
# setting = 'Exchange_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl128_dm8_nh2_el1_dl128_df1_fctimeF_ebTrue_dtExp_projection_0'  # 적절한 설정
# result_path = f'./results/{setting}/'

# preds = np.load(os.path.join(result_path, 'pred.npy'))
# trues = np.load(os.path.join(result_path, 'true.npy'))
# print("원본 정규화된 예측 shape:", preds.shape)

# # === 2. scaler 학습을 위한 train 데이터 불러오기 [CUSTOM] ===
# df_raw = pd.read_csv(os.path.join(root_path, data_path))
# cols = list(df_raw.columns)
# cols.remove(target)
# cols.remove('date')
# df_raw = df_raw[['date'] + cols + [target]]
# num_train = int(len(df_raw) * 0.7)

# # target과 날짜 열 제거 후 정렬
# if features in ['S', 'MS']:
#     df_data = df_raw[[target]]
# # target과 날짜 열 제거 후 정렬
# else:  # features == 'M'
#     df_data = df_raw[df_raw.columns[1:]]

# # train 기간
# train_data = df_data.iloc[0:num_train]

# scaler = StandardScaler()
# scaler.fit(train_data.values)

# # === 3. preds/trues 역정규화 ===
# shape = preds.shape
# preds_inverse = scaler.inverse_transform(preds.reshape(-1, shape[2])).reshape(shape)
# trues_inverse = scaler.inverse_transform(trues.reshape(-1, shape[2])).reshape(shape)

# # === 4. 저장 (선택 사항) ===
# np.save(os.path.join(result_path, 'pred_inverse.npy'), preds_inverse)
# np.save(os.path.join(result_path, 'true_inverse.npy'), trues_inverse)

# index = 700  # test 데이터에서 가장 앞쪽 시점의 예측 결과
# channel = -1

# pred = preds_inverse[index, :, channel]
# true = trues_inverse[index, :, channel]

# plt.figure(figsize=(15, 5))
# plt.plot(true, label='True')
# plt.plot(pred, label='Prediction')
# plt.title(f'Prediction vs True (index={index}, channel={channel}, inverse)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

######################################################### MS 데이터 inverse 하기 ##########################################
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import os

# # === 설정 ===
# root_path = './dataset/Appliances/'
# data_path = 'energy.csv'
# target = 'OT'   # 타겟 변수 이름
# features = 'MS'      # 'S', 'M', 'MS' 중 선택


# # === 1. pred.npy 파일 불러오기 ===
# setting = 'Appliances_96_720_iTransformer_custom_MS_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'  # 적절한 설정
# result_path = f'./results/{setting}/'

# preds = np.load(os.path.join(result_path, 'pred.npy'))
# trues = np.load(os.path.join(result_path, 'true.npy'))
# print("원본 정규화된 예측 shape:", preds.shape)

# # === 2. scaler 학습을 위한 train 데이터 불러오기 ===
# df_raw = pd.read_csv(os.path.join(root_path, data_path))
# cols = list(df_raw.columns)
# cols.remove(target)
# cols.remove('date')
# df_raw = df_raw[['date'] + cols + [target]]
# num_train = int(len(df_raw) * 0.7)

# # target과 날짜 열 제거 후 정렬
# if features in ['S', 'MS']:
#     df_data = df_raw[[target]]
# # target과 날짜 열 제거 후 정렬
# else:  # features == 'M'
#     df_data = df_raw[df_raw.columns[1:]]

# # train 기간: 앞 12개월
# train_data = df_data.iloc[0:num_train]

# scaler = StandardScaler()
# scaler.fit(train_data.values)

# # === 3. preds/trues 역정규화 ===
# shape = preds.shape
# preds_inverse = scaler.inverse_transform(preds.reshape(-1, shape[2])).reshape(shape)
# trues_inverse = scaler.inverse_transform(trues.reshape(-1, shape[2])).reshape(shape)

# # === 4. 저장 (선택 사항) ===
# np.save(os.path.join(result_path, 'pred_inverse.npy'), preds_inverse)
# np.save(os.path.join(result_path, 'true_inverse.npy'), trues_inverse)


# index = 0  # test 데이터에서 가장 앞쪽 시점의 예측 결과

# pred = preds_inverse[index, :]
# true = trues_inverse[index, :]

# plt.figure(figsize=(15, 5))
# plt.plot(true, label='True')
# plt.plot(pred, label='Prediction')
# plt.title(f'Prediction vs True (index={index}, inverse)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

################################################################ 애니메이션 M #####################################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# === 설정 ===
channel = -1

setting = 'train_FD001_24_96_iTransformer_CMAPSS_M_ft24_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'
result_path = f'./results/{setting}/'

preds = np.load(os.path.join(result_path, 'pred.npy'))
trues = np.load(os.path.join(result_path, 'true.npy'))

# === 애니메이션 설정 ===
fig, ax = plt.subplots(figsize=(12, 6))
line1, = ax.plot([], [], label='True')
line2, = ax.plot([], [], label='Prediction')
title = ax.set_title('')

metrics_text = ax.text(0.11, 0.95, '', transform=ax.transAxes, fontsize=12,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

ax.set_xlim(0, preds.shape[1])
ax.set_ylim(np.min(trues[:, :, channel]) * 0.7 , np.max(trues[:, :, channel]) * 0.9)
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.grid(True)
ax.legend()

def update(i):
    true_vals = trues[i, :, channel]
    pred_vals = preds[i, :, channel]

    line1.set_data(np.arange(preds.shape[1]), trues[i, :, channel])
    line2.set_data(np.arange(preds.shape[1]), preds[i, :, channel])
    title.set_text(f'Example {i}')

    mse = np.mean((true_vals - pred_vals)**2)
    mae = np.mean(np.abs(true_vals - pred_vals))
    metrics_text.set_text(f'MSE: {mse:.2f}\nMAE: {mae:.2f}')

    return line1, line2, title, metrics_text

ani = animation.FuncAnimation(fig, update, frames=500, interval=300, blit=True)
ani.save('prediction_vs_true_M.gif', writer='pillow', fps=5)

plt.tight_layout()
plt.show()

################################################################ 애니메이션 MS #####################################################################

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import os

# # === 설정 ===

# setting = 'Appliances_96_720_iTransformer_custom_MS_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'
# result_path = f'./results/{setting}/'

# preds = np.load(os.path.join(result_path, 'pred_inverse.npy'))
# trues = np.load(os.path.join(result_path, 'true_inverse.npy'))

# # === 애니메이션 설정 ===
# fig, ax = plt.subplots(figsize=(12, 6))
# line1, = ax.plot([], [], label='True')
# line2, = ax.plot([], [], label='Prediction')
# title = ax.set_title('')

# metrics_text = ax.text(0.11, 0.95, '', transform=ax.transAxes, fontsize=12,
#                        verticalalignment='top', horizontalalignment='right',
#                        bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

# ax.set_xlim(0, preds.shape[1])
# ax.set_ylim(np.min(trues[:, :]), np.max(trues[:, :]) * 1.1)
# ax.set_xlabel('Time Step')
# ax.set_ylabel('Value')
# ax.grid(True)
# ax.legend()

# def update(i):
#     true_vals = trues[i, :]
#     pred_vals = preds[i, :]

#     line1.set_data(np.arange(preds.shape[1]), trues[i, :])
#     line2.set_data(np.arange(preds.shape[1]), preds[i, :])
#     title.set_text(f'Example {i}')

#     mse = np.mean((true_vals - pred_vals)**2)
#     mae = np.mean(np.abs(true_vals - pred_vals))
#     metrics_text.set_text(f'MSE: {mse:.2f}\nMAE: {mae:.2f}')

#     return line1, line2, title, metrics_text

# ani = animation.FuncAnimation(fig, update, frames=500, interval=300, blit=True)
# ani.save('prediction_vs_true_MS.gif', writer='pillow', fps=5)

# plt.tight_layout()
# plt.show()