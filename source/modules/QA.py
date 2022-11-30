import sys
import os
sys.path.append(os.environ['source'])
import numpy as np

import torch
import torch.nn
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from module import Module
from utils import QA_DICT, get_intent_labels, get_slot_labels, load_tokenizer, compute_metrics
from data_loader import load_and_cache_examples, QADataset
from main import args




class QAModule(Module):
    def __init__(self, args):
        super().__init__(args)
        #define model
        config, _, model  = QA_DICT[args.model_type]
       
        self.config = config.from_pretrained(args.pretrained_model)
        self.model = model(
            self.config,
            args=args,
        )
        #define hparams
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.n_epochs = args.n_epochs
        self.device = args.device
        self.model = self.model.to(args.device)
    
    def train_step(self, batch):
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "token_type_ids": batch[2],
            "start": batch[3],
            "end": batch[4],
        }
        # if "distill" not in self.args.pretrained_model:
        #     inputs["token_type_ids"] = batch[2].to(self.device)
        outputs = self.model(**inputs)
        loss, (start_logits, end_logits) = outputs
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        return {"loss": loss, "start": start_logits, "end": end_logits}

    def eval_step(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start": batch[3],
                "end": batch[4],
            }

            outputs = self.model(**inputs)
            tmp_eval_loss, (start_logits, end_logits) = outputs
            loss = tmp_eval_loss.mean().item()


        return {"loss": loss, "start": start_logits, "end": end_logits}

if __name__ == "__main__":
    module = QAModule(args)
    tokenizer = load_tokenizer(args)
    dataset = QADataset(args, tokenizer)
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(module.train_step(batch))

    



        


        


        


