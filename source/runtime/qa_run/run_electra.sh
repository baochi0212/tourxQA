#additional 
freeze_weights = $1
python trainer.py  --pretrained --model_type electra --pretrained_model NlpHUST/electra-base-vn --device cuda --task QA --train_batch_size 16 --eval_batch_size 32 --tuning_metric F1_score --logging_step 1000
