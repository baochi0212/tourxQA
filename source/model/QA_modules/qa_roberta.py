import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchcrf import CRF
import pandas as pd

from transformers import AutoModel, AutoTokenizer, RobertaModel
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'

class QARoberta(nn.Module):
    def __init__(self, config, args):
        super().__init__()
        def CE_loss_fn(pred, label):
            loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, label)
            loss = torch.where(label != 0, loss , torch.tensor(0, dtype=torch.float).to(args.device))
            return loss.mean()
        self.bert_model = AutoModel.from_pretrained(args.pretrained_model)
        if args.freeze:
            print("Not retrained the model")
            for param in self.bert_model.parameters():
                param.requires_grad_(False)
        self.pretrained = args.pretrained
        self.linear = torch.nn.Linear(config.hidden_size, 2)
        self.relu = torch.nn.ReLU()
        self.loss_fn = CE_loss_fn
    
    def forward(self, input_ids, attention_mask, token_type_ids, start=None, end=None):
        outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
        logits = self.relu(self.linear(outputs))
        start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        if start is not None:
            loss = self.loss_fn(start_logits, start) + self.loss_fn(end_logits, end)  
            return loss, (start_logits, end_logits)
        else:
            return (start_logits, end_logits)