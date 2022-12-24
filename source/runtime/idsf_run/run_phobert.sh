#Distillation
#shared variables
export data="${data_dir}/IDSF/phoATIS"
#pretrained bert
python trainer.py --idsf_data_dir $data --task phoATIS_plus  --pretrained --model_type phobert --n_epochs 50 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 5e-5 --task phoATIS_plus 
python trainer.py --idsf_data_dir $data --from_pretrained_weights "./model_dir/idsf_weights/phobert_50_5e-05.pt" --pretrained --task phoATIS_plus  --model_type phobert --n_epochs 50 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 --task phoATIS_plus
python predict.py  --idsf_data_dir $data --pretrained --task phoATIS_plus  --model_type phobert --n_epochs 50 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 --task phoATIS_plus 

# #scratch bert
# python trainer.py --idsf_data_dir $data --task phoATIS_plus  --model_type phobert --n_epochs 30 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 5e-5 --task phoATIS_plus --use_intent_context_attention
# python trainer.py --idsf_data_dir $data --from_pretrained_weights "./model_dir/idsf_weights/phobert_30_5e-05.pt" --pretrained --task phoATIS_plus  --model_type phobert --n_epochs 30 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.15 --learning_rate 3e-5 --task phoATIS_plus --use_intent_context_attention
# python predict.py  --idsf_data_dir $data --task phoATIS_plus  --model_type phobert --n_epochs 30 --train_batch_size 32 --eval_batch_size 32 --rnn_num_layers 3  --device cuda --use_crf --logging_steps 140 --module_role IDSF  --intent_loss_coef 0.6 --learning_rate 3e-5 --task phoATIS_plus