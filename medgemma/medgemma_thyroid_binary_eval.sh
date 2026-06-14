python medgemma_thyroid_binary_eval.py \
  --model_dir /mnt/wangbd8/workspace/medgemma-4b-it \
  --image_dir /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/image \
  --label_json /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/finall_data/data_label.json \
  --out_csv medgemma_finall_data_Classification_all_preds.csv \
  --dtype bf16
