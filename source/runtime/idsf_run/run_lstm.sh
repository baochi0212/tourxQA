#GRU
#shared variables
export data="${data_dir}/IDSF/phoATIS"
python trainer.py --idsf_data_dir $data --task phoATIS  --model_type lstm --n_epochs 30 --train_batch_size 32 --eval_batch_size 2 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 5e-5 --task phoATIS_plus
python trainer.py --idsf_data_dir $data --from_pretrained_weights "./model_dir/idsf_weights/gru_30_5e-05.pt" --task phoATIS --model_type lstm --n_epochs 30 --train_batch_size 32 --eval_batch_size 2 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 --task phoATIS_plus
python predict.py  --idsf_data_dir $data --task phoATIS  --model_type lstm --n_epochs 30 --train_batch_size 32 --eval_batch_size 1 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 3e-5 --task phoATIS_plus