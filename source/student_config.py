import argparse
import os
import sys

working_dir = os.environ['source']
data_dir = os.environ['dir']
idsf_data_dir = os.environ['dir'] + '/data/processed/IDSF' 
qa_data_dir = os.environ['dir'] + '/data/processed/QA'


parser = argparse.ArgumentParser()

parser.add_argument('--text_question', type=str, default=None)
parser.add_argument('--text_question_log_dir', type=str, default=f'{data_dir}/deploy/api/log.txt')
parser.add_argument('--module_role', default='QA')
parser.add_argument('--from_pretrained_weights', default=None, type=str)
parser.add_argument('--qa_max_length', default=386, type=int)
parser.add_argument('--qa_log_dir', default='./model_dir/qa_weights')
parser.add_argument('--qa_model_dir', default='./model_dir/qa_weights')
parser.add_argument('--qa_data_dir', default=qa_data_dir, type=str)
parser.add_argument('--output_file', default='sample_output.txt', type=str)
parser.add_argument('--input_file', default='sample_input.txt', type=str)
parser.add_argument('--predict_task', default="test dataset", type=str)
parser.add_argument('--rnn_num_layers', default=3, type=int)
parser.add_argument('--model_type', default='distill-bert', type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--task', default='phoATIS', type=str)
parser.add_argument('--level', default='word-level', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--idsf_log_dir', default='./model_dir/idsf_weights')
parser.add_argument("--idsf_model_dir", default='./model_dir/idsf_weights', required=False, type=str, help="Path to save, load model")
parser.add_argument("--idsf_data_dir", default=idsf_data_dir+'/phoATIS', type=str, help="The input data dir")
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")
parser.add_argument("--freeze", type=int, default=0)
parser.add_argument("--tuning_metric", default="semantic_frame_acc", type=str, help="Metrics to tune when training")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
parser.add_argument(
    "--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization."
)
parser.add_argument("--learning_rate", default=5e-6, type=float, help="The initial learning rate for Adam.")
parser.add_argument(
    "--n_epochs", default=1, type=float, help="Total number of training epochs to perform."
)
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
)
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
parser.add_argument("--logging_steps", type=int, default=70, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=-1, help="Save checkpoint every X updates steps.")
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
parser.add_argument("--do_eval_dev", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
    "--ignore_index",
    default=0,
    type=int,
    help="Specifies a target value that is ignored and does not contribute to the input gradient",
)
parser.add_argument("--intent_loss_coef", type=float, default=1, help="Coefficient for the intent loss.")
parser.add_argument(
    "--token_level",
    type=str,
    default="word-level",
    help="Tokens are at syllable level or word level (Vietnamese) [word-level, syllable-level]",
)
parser.add_argument(
    "--early_stopping",
    type=int,
    default=5,
    help="Number of unincreased validation step to wait for early stopping",
)
parser.add_argument("--gpu_id", type=int, default=0, help="Select gpu id")
# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
# init pretrained
parser.add_argument("--pretrained", action="store_true", help="Whether to init model from pretrained base model")
parser.add_argument("--pretrained_model", default="vinai/phobert-base", type=str, help="The pretrained model path")
# Slot-intent interaction
parser.add_argument(
    "--use_intent_context_concat",
    action="store_true",
    help="Whether to feed context information of intent into slots vectors (simple concatenation)",
)
parser.add_argument(
    "--use_intent_context_attention",
    action="store_true",
    help="Whether to feed context information of intent into slots vectors (dot product attention)",
)
parser.add_argument(
    "--attention_embedding_size", type=int, default=200, help="hidden size of attention output vector"
)
parser.add_argument(
    "--slot_pad_label",
    default="PAD",
    type=str,
    help="Pad token for slot label pad (to be ignore when calculate loss)",
)
parser.add_argument(
    "--embedding_type", default="soft", type=str, help="Embedding type for intent vector (hard/soft)"
)
parser.add_argument("--use_attention_mask", action="store_true", help="Whether to use attention mask")
student_args = parser.parse_args()