from transformers import AutoTokenizer, AutoModel
from main import args
import torch.nn as nn
import torch
from torch.utils import data
MODEL_DICT = {'vinai/phobert-base': []}
class Module:
    def __init__(self, args):
        self.args = args
        #define modelling, example:
        # self.model, self.pretrained_path = MODEL_DICT[args.model_type] 
        #define hyper-params, example:
        # self.batch_size = args.batch_size
        # self.optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    def train_step(self, train_loader):
        pass
    def val_step(self, val_loader):
        pass 
    def test_step(self, test_loader):
        pass

    def loader(self, dataset, collate=False):
        pass 
    




    
