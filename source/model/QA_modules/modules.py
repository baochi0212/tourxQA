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
from dataset.test_dataset import IntentPOSDataset
import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset.test_dataset import IntentPOSDataset, QADataset
from utils.preprocess import get_label
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'

class QAModule(torch.nn.Module):
    def __init__(self, model_checkpoint, device, args=None, hidden=768, out=386):
        super().__init__()
        def CE_loss_fn(pred, label):
            loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, label)
            loss = torch.where(label != 0, loss , torch.tensor(0, dtype=torch.float).to(device))
            return loss.mean()
        self.bert_model = AutoModel.from_pretrained(model_checkpoint)
        self.pretrained = args.fast
        self.linear = torch.nn.Linear(hidden, 2)
        self.relu = torch.nn.ReLU()
        self.loss_fn = CE_loss_fn
    
    def forward(self, input_ids, attention_mask, start=None, end=None):
        if not self.pretrained:
            outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
            logits = self.relu(self.linear(outputs))
            start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
        else:
            outputs = self.bert_qa(input_ids=input_ids, attention_mask=attention_mask)
            start_logits, end_logits = outputs['start_logits'], outputs['end_logits']

        if start is not None:
          loss = self.loss_fn(start_logits, start) + self.loss_fn(end_logits, end)  
          return loss, {'start_logits':  start_logits, 'end_logits': end_logits}
        else:
          return {'start_logits':  start_logits, 'end_logits': end_logits}



      
 
# if __name__ == '__main__':
#     model_checkpoint = 'NlpHUST/bert-base-vn'
#     model = QAModule(model_checkpoint=model_checkpoint)
#     tokenizer_checkpoint = 'NlpHUST/bert-base-vn'
#     # model = RobertaModel.from_pretrained(model_checkpoint)
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
#     train_df = pd.read_csv(qa_processed + '/train.csv')
#     train_dataset = QADataset(train_df, tokenizer=tokenizer, mode='train')
#     train_loader = data.DataLoader(train_dataset, batch_size=32)
#     device = 'cpu'
#     batch = next(iter(train_loader))
#     b_input_ids, b_attn_mask, b_end, b_start = tuple(t.to(device) for t in batch)

#     input = tokenizer("DIT CON ME", return_tensors='pt',
#             max_length=258,
#             truncation="only_second",
#             padding="max_length",)
#     print(model(b_input_ids, b_attn_mask, b_start, b_end))

