export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# Daphnet Dataset
# Train: rows 0~9693 | Val: 9694~15000 | Test: 15001~end
# Sensor columns (9 features)
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv \
  --model_id Daphnet_seq96_pred96 \
  --model $model_name \
  --data Daphnet \
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
