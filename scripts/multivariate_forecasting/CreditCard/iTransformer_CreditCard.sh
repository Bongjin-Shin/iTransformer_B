export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# CreditCard Dataset
# Train: rows 0~4,920 | Val: 4,921~5,920 | Test: 5,921~end
# Sensor columns (29 features)
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 137_CreditCard_id_1_Finance_tr_500_1st_541.csv \
  --model_id CreditCard_seq96_pred96 \
  --model $model_name \
  --data CreditCard \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 29 \
  --dec_in 29 \
  --c_out 29 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
