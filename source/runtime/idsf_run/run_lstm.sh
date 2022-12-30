#GRU
#shared variables
export data="${data_dir}/IDSF/phoATIS"
python trainer.py --idsf_data_dir $data --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.5 --learning_rate 5e-5 
python predict.py --idsf_data_dir $data --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.5 --learning_rate 5e-5 

# python trainer.py --idsf_data_dir $data --from_pretrained_weights "./model_dir/idsf_weights/lstm_50_5e-05.pt"   --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 
# python predict.py  --idsf_data_dir $data   --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 

python trainer.py --idsf_data_dir $data --use_crf --task phoATIS_plus    --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.5 --learning_rate 5e-5 
python predict.py --idsf_data_dir $data --use_crf --task phoATIS_plus    --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.5 --learning_rate 5e-5 

# python trainer.py --idsf_data_dir $data --from_pretrained_weights "./model_dir/idsf_weights/lstm_50_5e-05.pt"   --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5  --use_crf
# python predict.py  --idsf_data_dir $data   --task phoATIS_plus  --model_type  lstm --n_epochs 50 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda    --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 --use_crf