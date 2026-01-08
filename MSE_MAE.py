import numpy as np
from utils.metrics import metric
preds_96 = np.load('./iTransformer/results/Spectral2_Appliances_96_96_Spectral2_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_96 = np.load('./iTransformer/results/Spectral2_Appliances_96_96_Spectral2_custom_M_ft96_sl48_ll96_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_192 = np.load('./iTransformer/results/Spectral2_Appliances_96_192_Spectral2_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_192 = np.load('./iTransformer/results/Spectral2_Appliances_96_192_Spectral2_custom_M_ft96_sl48_ll192_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_336 = np.load('./iTransformer/results/Spectral2_Appliances_96_336_Spectral2_custom_M_ft96_sl48_ll336_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_336 = np.load('./iTransformer/results/Spectral2_Appliances_96_336_Spectral2_custom_M_ft96_sl48_ll336_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
preds_720 = np.load('./iTransformer/results/Spectral2_Appliances_96_720_Spectral2_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/pred.npy')
trues_720 = np.load('./iTransformer/results/Spectral2_Appliances_96_720_Spectral2_custom_M_ft96_sl48_ll720_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dtExp_projection_0/true.npy')
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