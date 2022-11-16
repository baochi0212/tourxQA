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
parser.add_argument('--compare', default=False, type=bool)


def metrics(start, end, l_start, l_end, metrics='acc', input_ids=None, tokenizer=None, test=False, training=True):
  '''
  compare the pred with label ('l')
  '''
  if not test:
      start, end, l_start, l_end = start.view(-1), end.view(-1), l_start.view(-1), l_end.view(-1)
      count = 0
      if metrics == 'acc':
        for i in range(start.shape[0]):
          if start[i] == l_start[i] and end[i] == l_end[i]:
             count += 1 
        return count/start.shape[0]
  else:
      start, end = start.view(-1), end.view(-1)
      l_start, l_end =  l_start.view(start.shape[0], -1), l_end.view(end.shape[0], -1)
      count = 0
      if metrics == 'acc':
        for i in range(start.shape[0]):
            start_end = (start[i].item(), end[i].item())
            l_start_end  = [(m.item(), n.item()) for m, n in zip(l_start[i], l_end[i])]
            if start_end in l_start_end:
                count += 1 
            elif not training:
                print("MISTAKES", start_end, l_start_end)
                print("CONTEXT and QUESTION", tokenizer.decode(input_ids[i]))
                print("ANSWERS", tokenizer.decode(input_ids[i][l_start_end[0][0]:l_start_end[0][1]+1]))
                print("PREDS", tokenizer.decode(input_ids[i][start_end[0]:start_end[1]+1]))
        #         if count % 10 == 0:
        #             # print("SAMPLE", start_end, l_start_end)
        # # print("COUNT:", count)
        return count/start.shape[0]

def evaluate_QA(model, val_dataloader, device, tokenizer, print_fn=False, test=False, pipeline=False, training=True):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    count = 0
    with torch.no_grad():
      #if using the pipeline API of huggingface
      if not pipeline:
          for batch in val_dataloader:
                # print(batch)
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_start, b_end = tuple(t.to(device) for t in batch)
                if len(b_start.shape) == 2:
                    b_start_sub, b_end_sub = b_start[:, 0], b_end[:, 0]
                else:
                    b_start_sub, b_end_sub = b_start, b_end
                if not test:
                    loss, outputs  = model(b_input_ids, b_attn_mask, b_start_sub, b_end_sub)
                    val_loss.append(loss.item())
                    count += 1 
                else:

                    outputs = model(b_input_ids, b_attn_mask)
                    #double check
                    count += 1
                start_logits = outputs['start_logits']
                end_logits  = outputs['end_logits']
                start, end = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)
                # if count % 3 == 0:
                #     print("Q and C: ", tokenizer.decode(b_input_ids[0]))
                #     print("answers:", tokenizer.decode(b_input_ids[0][start[0]:end[0]+1]))
                # in squad-v2 val is same as train (1-label) but in Vi-squad it's multi-label
                val_accuracy.append(metrics(start, end, b_start, b_end, tokenizer=tokenizer, input_ids=b_input_ids, metrics='acc', test=True, training=training))
          val_loss =  np.array(val_loss).mean()
          val_accuracy = np.array(val_accuracy).mean()
    return val_loss, val_accuracy


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
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    model_path = './models/weights/model.pt' 

    test_df = pd.read_csv(qa_processed + '/dev.csv') if mode == 'test' else pd.read_csv(qa_processed + '/dev.csv')
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test', MAX_LENGTH=max_length)
    test_loader = data.DataLoader(test_dataset, batch_size=1)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    model.eval()
    print(evaluate_QA(model.to(device), test_loader, tokenizer=tokenizer, device=device, test=True, training=False))


