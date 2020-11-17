CUDA_VISIBLE_DEVICES=4,5,6,7 \
python train.py \
  --model_config config/model_config_small.json \
  --tokenized_data_path data/tokenized/ \
  --tokenizer_path cache/vocab_small.txt \
  --raw_data_path data/train_ci.json \
  --epochs 300 \
  --output_dir model/ \
  --device 0,1,2,3 \
  --batch_size 8 \
  --raw \
  --min_length 1 \
  --pretrained_model ./pretrain_model/sanwen_shi_model/ \
  --lr 3e-4


