export lr=5e-5
export c=0.5
export e=50
export s=333
echo "${lr}"
export MODEL_DIR=JointIDSF_PhoBERTencoder
export MODEL_DIR=$MODEL_DIR"/"$lr"/"$c"/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir PhoATIS \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 140 \
                  --logging_steps 140 \
                  --num_train_epochs $e\
                  --tuning_metric loss \
                  --use_crf \
                  --gpu_id 0 \
                  --intent_loss_coef $c \
                  --learning_rate $lr \
                  --use_crf \
                  
