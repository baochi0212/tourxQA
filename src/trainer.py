import torch
import torch.nn 
import sys
import os
from utils.utils import MODEL_MAP, PRETRAINED_MAP, CONFIG_MAP
#add source directory 
src_dir = os.environ['src']
sys.path.append(src_dir)

class Trainer:
    '''
    our trainer for multi tasking
    - model load: task + pretrained type

    '''
    def __init__(self, args, train_dataset, test_dataset, val_dataset):
        '''
        load datasets, models, fine tuning or load trained models
        '''
        self.args = args
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        model = MODEL_MAP[args.task]
        config, tokenizer = CONFIG_MAP[args.model_type]
        pretrained_path = PRETRAINED_MAP[args.model_type]
        if args.pretrained: 
            self.model = model(config).load_state_dict(torch.load(args.pretrained_path))
            self.tokenizer = tokenizer.from_pretrained(pretrained_path)
        else:
            self.model = model(config).from_pretrained(pretrained_path)
            self.tokenizer = tokenizer.from_pretrained(pretrained_path)
        
    def train(self):
        