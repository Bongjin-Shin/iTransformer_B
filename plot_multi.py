import numpy as np
import matplotlib.pyplot as plt
import os

# 실험 세팅 이름 (run.py에서 setting 문자열과 동일하게)
setting = 'Appliances_96_720_iTransformer_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0'  # 예: 'iTransformer_ETTh1_M_96_48_96_...' 형식

# 저장된 결과 파일 경로
result_folder = f'./results/{setting}/'
preds = np.load(os.path.join(result_folder, 'pred.npy'))
trues = np.load(os.path.join(result_folder, 'true.npy'))

# 차원 확인 (B, L, D): 배치 수, 예측 길이, 변수 수
print('preds shape:', preds.shape)
print('trues shape:', trues.shape)

# === 시각화 ===
# index = 2784  # 보고 싶은 배치 인덱스 선택
channel = 0  # 변수 인덱스 선택


for i in range(10):
    plt.figure(figsize=(12, 6))
    plt.plot(trues[i, :, channel], label='True')
    plt.plot(preds[i, :, channel], label='Pred')
    plt.title(f'Example {i}')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()