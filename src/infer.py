'''
for get sample prediction
'''

import random
import time
import numpy as np
from tqdm.auto import tqdm
import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from accelerate import Accelerator
from torchcrf import CRF 
import pandas as pd 

import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset.test_dataset import IntentPOSDataset, QADataset
from models.modules import IntentPOSModule, CRFPOS, CustomConfig, QAModule
from utils.preprocess import get_label
import argparse
import logging

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
tokenizer_checkpoint = 'NlpHUST/bert-base-vn'

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=9, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--learning_rate', default=3e-5, type=float, help="Learning rate")
parser.add_argument('--pretrained_model', default='NlpHUST/bert-base-vn', type=str)
parser.add_argument('--pretrained_input', default=768, type=int)
parser.add_argument('--predict_mode', default='test', type=str)
parser.add_argument('--max_length', default=500, type=int)
parser.add_argument('--compare', action='store_true', default=False)


def predict(model, tokenizer, file='sample_input.txt', out='sample_output.txt', MAX_LENGTH=386):
    question = []
    context = []
    with open(file, 'r') as f:
        for line in f.readlines():
            q, c = line.strip().split("$$$")
            question.append(q.strip())
            context.append(c.strip())
    with open(out, 'w') as f:
        for q, c in zip(question, context):
            input = tokenizer(q, c, return_tensors='pt',
                max_length=MAX_LENGTH,
                truncation="only_second",
                return_offsets_mapping=True,
                padding="max_length")
            outputs = model(input.input_ids, input.attention_mask)

            start_logits = outputs['start_logits'][0]
            end_logits  = outputs['end_logits'][0]
            start, end = torch.argmax(start_logits, -1).item(), torch.argmax(end_logits, -1).item()
            output_string = tokenizer.decode(input.input_ids[start:end+1])
            f.write(q + "FROM" + c)
            f.write(str(start) + "TO" + str(end))
            f.write(output_string)
        

if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size
    mode = args.predict_mode
    max_length = args.max_length
    #for comparison with SOTA
    checkpoint = 'nguyenvulebinh/vi-mrc-large'

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = args.pretrained_model

    if not args.compare:
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model = QAModule(model_checkpoint=model_checkpoint, device=device, hidden=args.pretrained_input).to(device)
        model_path = './models/weights/model.pt'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
    else:
        print("------------USING THE PRETRAINED-----------")
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
     



    model.eval()
    predict(model, tokenizer)



