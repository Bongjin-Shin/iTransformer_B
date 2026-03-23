export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# Genesis Dataset
# Train: rows 0~4,055 | Val: 4,056~6,000 | Test: 6,001~end
# Sensor columns (18 features)
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv \
  --model_id Genesis_seq96_pred96 \
  --model $model_name \
  --data Genesis \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 18 \
  --dec_in 18 \
  --c_out 18 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
