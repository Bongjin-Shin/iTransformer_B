export CUDA_VISIBLE_DEVICES=1

model_name=iTransformer

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Appliances/ \
  --data_path energy.csv \
  --model_id Appliances_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Appliances/ \
  --data_path energy.csv \
  --model_id Appliances_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Appliances/ \
  --data_path energy.csv \
  --model_id Appliances_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Appliances/ \
  --data_path energy.csv \
  --model_id Appliances_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 26 \
  --dec_in 26 \
  --c_out 26 \
  --des 'Exp' \
  --d_model 512\
  --d_ff 512\
  --itr 1