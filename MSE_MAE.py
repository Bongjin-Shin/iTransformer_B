# import numpy as np
# from utils.metrics import metric          # 같은 repo에 이미 있음

# # (1) MS 실험 결과
# ms_metrics = np.load('./results/ETTh1_96_96_iTransformer_ETTh1_MS_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/metrics.npy')
# print('MS (single-target)  MAE, MSE, RMSE, MAPE, MSPE:', ms_metrics)

# # (2) M 실험 prediction / ground-truth 로드
# preds = np.load('./results/ETTh1_96_96_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
# trues = np.load('./results/ETTh1_96_96_iTransformer_ETTh1_M_ft96_sl48_ll96_pl256_dm8_nh2_el1_dl256_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')

# # (3) 타깃 변수 인덱스 찾기
# import pandas as pd
# cols = pd.read_csv('dataset/ETT-small/ETTh1.csv', nrows=0).columns[1:]  # date 제외
# target_idx = list(cols).index('OT')             # 예: OT가 마지막이면 6

# # (4) 타깃만 슬라이싱 & 지표 계산
# pred_t = preds[..., target_idx:target_idx+1]    # shape 유지
# true_t = trues[..., target_idx:target_idx+1]
# mae, mse, rmse, mape, mspe = metric(pred_t, true_t)
# print('M (multi-target) 의 타깃 OT 지표:', mae, mse, rmse, mape, mspe)

import numpy as np
from utils.metrics import metric
preds_96 = np.load('./results/Appliances_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_96 = np.load('./results/Appliances_96_96_iTransformer_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_192 = np.load('./results/Appliances_96_192_iTransformer_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_192 = np.load('./results/Appliances_96_192_iTransformer_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_336 = np.load('./results/Appliances_96_336_iTransformer_custom_M_ft96_sl48_ll336_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_336 = np.load('./results/Appliances_96_336_iTransformer_custom_M_ft96_sl48_ll336_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_720 = np.load('./results/Appliances_96_720_iTransformer_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_720 = np.load('./results/Appliances_96_720_iTransformer_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
mae_96, mse_96, rmse_96, mape_96, mspe_96 = metric(preds_96, trues_96)
mae_192, mse_192, rmse_192, mape_192, mspe_192 = metric(preds_192, trues_192)
mae_336, mse_336, rmse_336, mape_336, mspe_336 = metric(preds_336, trues_336)
mae_720, mse_720, rmse_720, mape_720, mspe_720 = metric(preds_720, trues_720)

mae = []
mse = []
mae.append(mae_96)
mae.append(mae_192)
mae.append(mae_336)
mae.append(mae_720)
mse.append(mse_96)
mse.append(mse_192)
mse.append(mse_336)
mse.append(mse_720)

mae_avg = sum(mae) / len(mae)
mse_avg = sum(mse) / len(mse)

print(f'MSE : {mse_avg}, MAE : {mae_avg}')
print(f'MSE_96 : {mse_96}, MAE_96 : {mae_96}')
print(f'MSE_192 : {mse_192}, MAE_192 : {mae_192}')
print(f'MSE_336 : {mse_336}, MAE_336 : {mae_336}')
print(f'MSE_720 : {mse_720}, MAE_720 : {mae_720}')