import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from torchcrf import CRF

from transformers import AutoModel, AutoTokenizer
from dataset.test_dataset import IntentPOSDataset
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
class CustomConfig:
    def __init__(self, n_intent, n_pos, embedding_size=768):
        self.embedding_size = embedding_size
        self.n_intent = n_intent
        self.n_pos = n_pos

class IntentPOSModule(nn.Module):
    '''
    - embedding:
    - intent head:
    - POS head:
    - config
    '''
    def __init__(self, config):
        super().__init__()
        self.embedding = AutoModel.from_pretrained("vinai/phobert-base")
        self.intent_head = nn.Linear(config.embedding_size, config.n_intent)
        self.pos_head = nn.Linear(config.embedding_size, config.n_pos)
    def forward(self, input, mask):
        x = self.embedding(input_ids=input, attention_mask=mask)
        crf_pos = nn.functional.relu(self.pos_head(x['last_hidden_state']))
        return torch.sigmoid(nn.functional.relu(self.intent_head(x['last_hidden_state'].mean(dim=1)))), nn.functional.relu(self.pos_head(x['last_hidden_state']))

class CRFPOS(nn.Module):
    '''
    - embedding:
    - intent head:
    - POS head:
    - config
    '''
    def __init__(self, config):
        super().__init__()
        self.embedding = AutoModel.from_pretrained("vinai/phobert-base")
        self.intent_head = nn.Linear(config.embedding_size, config.n_intent)
        self.pos_head = nn.Linear(config.embedding_size, config.n_pos)
        self.CRF = CRF(config.n_pos)
    def forward(self, input, pos_label, mask):
        x = self.embedding(input_ids=input, attention_mask=mask)
        crf_pos = nn.functional.relu(self.pos_head(x['last_hidden_state']))
        return torch.sigmoid(nn.functional.relu(self.intent_head(x['last_hidden_state'].mean(dim=1)))), crf_pos, -self.CRF(crf_pos.permute(1, 0, 2), pos_label.permute(1, 0))
class QAModule(torch.nn.Module):
  def __init__(self, model_checkpoint, hidden=768, out=386):
    super().__init__()
    def CE_loss_fn(pred, label):
    #     print("pred", pred.shape)
        loss = torch.nn.CrossEntropyLoss(reduction='none')(pred, label)
        loss = torch.where(label != 0, loss, torch.tensor(0, dtype=torch.float).to(device))
        return loss.mean()
#     self.bert_qa = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint).cuda()
    self.bert_model = AutoModel.from_pretrained(model_checkpoint)
    self.linear = torch.nn.Linear(hidden, 2)
    self.relu = torch.nn.ReLU()
    self.loss_fn = CE_loss_fn
   
  def forward(self, input_ids, attention_mask, start=None, end=None ):
    print("INPUT")
    outputs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
    print("OUTPUS", outputs.shape)
#     print("outputs", outputs.shape)
#     start_logits, end_logits = outputs['start_logits'], outputs['end_logits']
    logits = self.relu(self.linear(outputs))
    
    start_logits, end_logits = logits[:, :, 0], logits[:, :, 1]
    print("XXXXX", start_logits.shape, start.shape)
    if start is not None:
      loss = self.loss_fn(start_logits, start) + self.loss_fn(end_logits, end)
#       return loss, outputs
      return loss, {'start_logits':  start_logits, 'end_logits': end_logits}
    else:
      return start_logits, end_logits



      
 
if __name__ == '__main__':
    dataset = IntentPOSDataset(raw_dir, MAX_LENGTH=30)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    config = CustomConfig(n_pos=dataset.n_pos, n_intent=dataset.n_intent)
    # net = IntentPOSModule(config)
    net = CRFPOS(config)
    sample = next(iter(dataloader))
    # print("sample: ", sample[0]['input_ids'].shape, sample[0]['attention_mask'].shape)
    # print("test output: ", net(sample[0], sample[1])[0].shape, net(sample[0], sample[1])[1].shape, sample[1].shape, sample[2].shape)
    print("test output: ", net(sample[0], sample[-1], sample[1])[2])

