export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer
data_name=NPY
root_path=./dataset/MSL_SMAP/

# ============================================================
# 반복 실험 스크립트
# train 폴더 내의 A-*.npy 파일을 모두 찾아서 순차적으로 실행
# ============================================================

for file_path in $(ls ${root_path}train/E-*.npy | sort -V); do
    # 1. 파일명 추출 (예: dataset/train/A-1.npy -> A-1.npy)
    filename=$(basename "$file_path")
    
    # 2. 확장자 제거한 이름 (예: A-1.npy -> A-1) -> 모델 ID로 사용
    file_id="${filename%.*}"
    
    echo "========================================================"
    echo " Start Training & Testing for: $filename "
    echo "========================================================"

    # 3. 학습 및 평가 실행
    # --data_path에 'A-1.npy' 전체 이름을 넘겨줍니다.
    python -u run.py \
      --is_training 1 \
      --root_path $root_path \
      --data_path $filename \
      --model_id ${file_id}_seq96_pred96 \
      --model $model_name \
      --data $data_name \
      --features S \
      --seq_len 96 \
      --label_len 48 \
      --pred_len 96 \
      --e_layers 3 \
      --enc_in 1 \
      --dec_in 1 \
      --c_out 1 \
      --des 'Exp' \
      --d_model 512 \
      --d_ff 512 \
      --itr 1

done