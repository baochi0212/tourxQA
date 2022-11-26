import torch
import torch.nn as nn
from torch.utils import data 


from train import Trainer
    
    
    
class Trainer:
    def __init__(self, args):
        #argument
        self.args = args
        #model config
        if args.pretrained:

    
    def fit(self, train_dataset, val_dataset):
        pass

    def eval(self):
        pass

    def predict(self):
        pass
    
    
    
    
    
    def save(self):
        model_path = args.model_path
        torch.save(self.model.state_dict(), model_path)

    def load(self):
        self.model = self.model.load_state_dict(torch.load(args.model_path))

    def configure_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        t_total = len(train_l)
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        return {'optimizer': self.optimizer, 'scheduler': self.scheduler}