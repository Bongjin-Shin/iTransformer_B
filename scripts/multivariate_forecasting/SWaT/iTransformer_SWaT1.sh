export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

# ============================================================
# SWaT_id_1 Dataset
# Train: rows 0~3,749 | Val: 3,750~5,500 | Test: 5,501~end
# Sensor columns (66 features)
# ============================================================

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/TSB-AD-M/ \
  --data_path 171_SWaT_id_1_Sensor_tr_3749_1st_9522.csv \
  --model_id SWaT1_seq96_pred96 \
  --model $model_name \
  --data SWaT1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 66 \
  --dec_in 66 \
  --c_out 66 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --itr 1
