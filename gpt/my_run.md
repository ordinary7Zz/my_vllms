set OPENAI_API_KEY=你的key

rem 1) Thyroid ultrasound benign-vs-malignant classification
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\read_500\BM\images" ^
  --label_json "D:\WorkFiles\ThyroidAgent\read_500\BM\500_TestData_Malignancy_Cls.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gpt5.5_BM_preds2.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\TN5K\test\images_500" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\TN5K\test\TN5K_test_label_500.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gpt5.5_TN5K_preds.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\ThyroidXL\test\images_500" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\ThyroidXL\test\ThyroidXL_test_label_500.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gpt5.5_ThyroidXL_preds.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\DDTI_Classification\all\images_processed" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\DDTI_Classification\all\DDTI_Classification_test_label.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gpt5.5_DDTI_preds.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\read_500\BM\images" ^
  --label_json "D:\WorkFiles\ThyroidAgent\read_500\BM\500_TestData_Malignancy_Cls.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gemini3.1_BM_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\TN5K\test\images_300" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\TN5K\test\TN5K_test_label_300.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gemini3.1_TN5K_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\ThyroidXL\test\images_300" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\ThyroidXL\test\ThyroidXL_test_label_300.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gemini3.1_ThyroidXL_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\DDTI_Classification\all\images_processed_150" ^
  --label_json "D:\WorkFiles\A DataSets\ThyroidAgent\train_val_test\DDTI_Classification\all\DDTI_Classification_test_label_150.json" ^
  --task "malignancy" ^
  --label_key "malignancy" ^
  --out_json ".\outputs\BM\gemini3.1_DDTI_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

rem 2) Thyroid ultrasound cervical lymph node metastasis classification
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\Lymph Node Metastasis\Lymph_Node_Metastasis\center2\images" ^
  --label_json "D:\WorkFiles\ThyroidAgent\Lymph Node Metastasis\Lymph_Node_Metastasis\LymphUs_test_labels.json" ^
  --task "lymph_node_metastasis" ^
  --label_key "LNM_CN01" ^
  --out_json ".\outputs\LNM\gemini_thyroid_lymph_node_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\Lymph Node Metastasis\Lymph_Node_Metastasis\center2\images" ^
  --label_json "D:\WorkFiles\ThyroidAgent\Lymph Node Metastasis\Lymph_Node_Metastasis\LymphUs_test_labels.json" ^
  --task "lymph_node_metastasis" ^
  --label_key "LNM_CN01" ^
  --out_json ".\outputs\LNM\gpt_thyroid_lymph_node_preds.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

rem 3) Thyroid ultrasound FTC-vs-PTC classification
python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\FangDai_Thyroid_Ultrasound_Images_cropped" ^
  --label_json "D:\WorkFiles\ThyroidAgent\FangDai_Thyroid_Ultrasound_Images_cropped\FangDai_train_labels.json" ^
  --task "ftc_ptc" ^
  --label_key "FTCPTC" ^
  --out_json ".\outputs\FTCPTC\gemini_thyroid_ftc_ptc_preds.json" ^
  --model "gemini-3.1-pro" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0

python gpt_thyroid_binary_eval.py ^
  --image_dir "D:\WorkFiles\ThyroidAgent\FangDai_Thyroid_Ultrasound_Images_cropped" ^
  --label_json "D:\WorkFiles\ThyroidAgent\FangDai_Thyroid_Ultrasound_Images_cropped\FangDai_train_labels.json" ^
  --task "ftc_ptc" ^
  --label_key "FTCPTC" ^
  --out_json ".\outputs\FTCPTC\gpt_thyroid_ftc_ptc_preds.json" ^
  --model "gpt-5.5" ^
  --base_url "https://api.poe.com/v1" ^
  --threshold 0.5 ^
  --ci_bootstrap 2000 ^
  --ci_alpha 0.95 ^
  --max_retries 3 ^
  --retry_delay 2.0