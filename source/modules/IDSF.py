import sys
import os
sys.path.append(os.environ['source'])
import numpy as np

import torch
import torch.nn
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup
from .module import Module
from utils import MODEL_DICT, get_intent_labels, get_slot_labels, load_tokenizer, compute_metrics
# from data_loader import load_and_cache_examples
# from main import args




class ISDFModule(Module):
    def __init__(self, args):
        super().__init__(args)
        #define model
        config, _, model  = MODEL_DICT[args.model_type]
        self.intent_label_lst, self.slot_label_lst = get_intent_labels(args), get_slot_labels(args)
       
        if args.pretrained:
            self.config = config.from_pretrained(args.pretrained_model)
            self.model = model.from_pretrained(
                args.pretrained_model,
                args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,
            )
        else:
           
            self.config = config.from_pretrained(args.pretrained_model)
            self.model = model(self.config, args=args,
                intent_label_lst=self.intent_label_lst,
                slot_label_lst=self.slot_label_lst,)
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
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        if "distill" not in self.args.pretrained_model:
            inputs["token_type_ids"] = batch[2].to(self.device)
        outputs = self.model(**inputs)
        loss = outputs[0]
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        return {"loss": loss}

    def eval_step(self, batch):
        batch = tuple(t.to(self.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "intent_label_ids": batch[3],
                "slot_labels_ids": batch[4],
            }
            if "bert" not in self.args.model_type:
                inputs["token_type_ids"] = batch[2].to(self.device)
            outputs = self.model(**inputs)
            tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
            loss = tmp_eval_loss.mean().item()


        return {"loss": loss, "intent": intent_logits, "slot": slot_logits, "inputs": inputs}

if __name__ == "__main__":
    module = ISDFModule(args)
    tokenizer = load_tokenizer(args)
    dataset = load_and_cache_examples(args, tokenizer, mode="train")
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(module.train_step(batch))

    



        


        


        


