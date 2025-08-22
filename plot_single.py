import numpy as np
import matplotlib.pyplot as plt
import os

# === 설정 ===
setting = 'traffic_96_192_iTransformer_custom_M_ft96_sl48_ll192_pl512_dm8_nh4_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'  # 실험에 맞게 수정
result_path = f'./results/{setting}/'

# === 데이터 불러오기 ===
preds = np.load(os.path.join(result_path, 'pred.npy'))
trues = np.load(os.path.join(result_path, 'true.npy'))

print('Prediction shape:', preds.shape)  # (batch, pred_len, channel)
print('True shape:', trues.shape)

# === 시각화 ===
index = 0  # 보고 싶은 배치 인덱스 선택
channel = 0  # 변수 인덱스 선택

pred = preds[index, :, channel]
true = trues[index, :, channel]

plt.figure(figsize=(15, 7))
plt.plot(true, label='True')
plt.plot(pred, label='Prediction')
plt.title(f'Prediction vs True (index={index}, channel={channel})')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(f'{result_path}/compare_index{index}_channel{channel}.png')
plt.show()
