export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# 1. Dataset_CMAPSS: 일반 슬라이딩 윈도우 방식
# Train/Val: 엔진 1~70 / 71~80, 각 100행까지 (정상 패턴 학습)
# Test: 엔진 81~100, 각 엔진 전체 데이터 (정상→비정상 구간 평가)
# 제외 센서: s1, s5, s6, s10, s16, s18, s19 → 사용 센서: 14개
# ============================================================

# 1-1. seq_len=12, pred_len=48
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/CMAPSSData/ \
  --data_path train_FD001_with_columns.csv \
  --model_id 100_train_FD001_seq12_pred48 \
  --model $model_name \
  --data CMAPSS \
  --features M \
  --seq_len 12 \
  --label_len 6 \
  --pred_len 48 \
  --e_layers 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_limit 100 \
  --itr 1

# 1-2. seq_len=24, pred_len=48
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/CMAPSSData/ \
  --data_path train_FD001_with_columns.csv \
  --model_id 100_train_FD001_seq24_pred48 \
  --model $model_name \
  --data CMAPSS \
  --features M \
  --seq_len 24 \
  --label_len 12 \
  --pred_len 48 \
  --e_layers 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_limit 100 \
  --itr 1

# 1-3. seq_len=48, pred_len=48
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/CMAPSSData/ \
  --data_path train_FD001_with_columns.csv \
  --model_id 100_train_FD001_seq48_pred48 \
  --model $model_name \
  --data CMAPSS \
  --features M \
  --seq_len 48 \
  --label_len 24 \
  --pred_len 48 \
  --e_layers 3 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 14 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --train_limit 100 \
  --itr 1

# ============================================================
# 2-1. seq_len=12, pred_len=12
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/CMAPSSData/ \
#   --data_path train_FD001_with_columns.csv \
#   --model_id train_FD001_seq12_pred48 \
#   --model $model_name \
#   --data CMAPSS \
#   --features M \
#   --seq_len 12 \
#   --label_len 6 \
#   --pred_len 12 \
#   --e_layers 3 \
#   --enc_in 14 \
#   --dec_in 14 \
#   --c_out 14 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --train_limit 100 \
#   --itr 1

# # 2-2. seq_len=24, pred_len=12
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/CMAPSSData/ \
#   --data_path train_FD001_with_columns.csv \
#   --model_id train_FD001_seq24_pred48 \
#   --model $model_name \
#   --data CMAPSS \
#   --features M \
#   --seq_len 24 \
#   --label_len 12 \
#   --pred_len 12 \
#   --e_layers 3 \
#   --enc_in 14 \
#   --dec_in 14 \
#   --c_out 14 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --train_limit 100 \
#   --itr 1

# # 2-3. seq_len=48, pred_len=12
# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/CMAPSSData/ \
#   --data_path train_FD001_with_columns.csv \
#   --model_id train_FD001_seq48_pred48 \
#   --model $model_name \
#   --data CMAPSS \
#   --features M \
#   --seq_len 48 \
#   --label_len 24 \
#   --pred_len 12 \
#   --e_layers 3 \
#   --enc_in 14 \
#   --dec_in 14 \
#   --c_out 14 \
#   --des 'Exp' \
#   --d_model 512 \
#   --d_ff 512 \
#   --train_limit 100 \
#   --itr 1
