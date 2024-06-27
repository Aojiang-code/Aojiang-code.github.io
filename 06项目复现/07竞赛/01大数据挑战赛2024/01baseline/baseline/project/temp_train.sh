export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

python -u train.py \
  --root_path ./dataset/global \
  --data_path temp.npy \
  --model_id v1 \
  --model $model_name \
  --data Meteorology \
  --features MS \
  --seq_len 168 \
  --label_len 1 \
  --pred_len 24 \
  --e_layers 1 \
  --enc_in 37 \
  --d_model 64 \
  --d_ff 64 \
  --n_heads 1 \
  --des 'global_temp' \
  --learning_rate 0.01 \
  --batch_size 40960 \
  --train_epochs 1