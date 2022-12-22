#additional 
export freeze_weights=$1
python trainer.py --freeze $freeze_weights --pretrained --model_type xlm-roberta-base --pretrained_model xlm-roberta-base --device cuda --task QA --train_batch_size 16 --eval_batch_size 32 --tuning_metric F1_score --logging_step 500