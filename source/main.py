import argparse
import os
import sys

working_dir = os.environ['source']
data_dir = os.environ['dir'] + '/data/raw' 

parser = argparse.ArgumentParser()
parser.add_argument('--rnn_num_layers', default=9, type=int)
parser.add_argument('--model_type', default='vinai/phobert-base', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--task', default='phoATIS', type=str)
parser.add_argument('--level', default='word-level', type=str)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument("--model_dir", default='./model_dir', required=False, type=str, help="Path to save, load model")
parser.add_argument("--data_dir", default=data_dir+'/PhoATIS', type=str, help="The input data dir")
parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

parser.add_argument("--tuning_metric", default="loss", type=str, help="Metrics to tune when training")
parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size for training.")
parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
parser.add_argument(
    "--max_seq_len", default=50, type=int, help="The maximum total input sequence length after tokenization."
)
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument(
    "--n_epochs", default=10.0, type=float, help="Total number of training epochs to perform."
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
parser.add_argument("--logging_steps", type=int, default=350, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
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
parser.add_argument("--intent_loss_coef", type=float, default=0.5, help="Coefficient for the intent loss.")
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
args = parser.parse_args()