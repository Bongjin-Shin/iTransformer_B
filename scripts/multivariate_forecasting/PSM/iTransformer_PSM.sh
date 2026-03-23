export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# PSM Dataset
# Train: rows 0~50,000 | Val: 50,001~60,000 | Test: 60,001~end
# Sensor columns (25): feature_0 ~ feature_24
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 115_PSM_id_1_Facility_tr_50000_1st_129872.csv \
  --model_id PSM_seq96_pred96 \
  --model $model_name \
  --data PSM \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 25 \
  --dec_in 25 \
  --c_out 25 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
