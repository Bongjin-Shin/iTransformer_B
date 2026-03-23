export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# GECCO Water Quality Sensor Dataset
# Train: rows 0~12,000 | Val: 12,001~15,000 | Test: 15,001~end
# Sensor columns (9): Tp, Cl, pH, Redox, Leit, Trueb, Cl_2, Fm, Fm_2
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv \
  --model_id GECCO_seq96_pred96 \
  --model $model_name \
  --data GECCO \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
