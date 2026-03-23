export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# SWaT_id_2 Dataset
# Train: rows 0~23,700 | Val: 23,701~33,700 | Test: 33,701~end
# Sensor columns (51 features)
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 172_SWaT_id_2_Sensor_tr_23700_1st_23800.csv \
  --model_id SWaT2_seq96_pred96 \
  --model $model_name \
  --data SWaT2 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 51 \
  --dec_in 51 \
  --c_out 51 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
