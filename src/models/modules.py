import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch
import torch.nn as nn
import torchvision
from torch.utils import data

from transformers import AutoModel, AutoTokenizer
from dataset.test_dataset import IntentPOSDataset
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
class CustomConfig:
    def __init__(self, n_intent, n_pos, embedding_size=1024):
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

    def forward(self, x):
        x = self.embedding(x)
        print(x)
        # return self.intent_head(x['last']), self.pos_head()

if __name__ == '__main__':
    dataset = IntentPOSDataset(raw_dir)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    config = CustomConfig(n_pos=dataset.n_pos, n_intent=dataset.n_intent)
    net = IntentPOSModule(config)
    sample = next(iter(dataloader))
    # print("sample: ", sample[0]['input_ids'].shape, sample[0]['attention_mask'].shape)
    print("test output: ", net(sample[0]).shape)

