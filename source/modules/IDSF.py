import sys
import os
sys.path.append(os.environ['source'])
import numpy as np

import torch
import torch.nn
from torch.utils import data
from transformers import AdamW, get_linear_schedule_with_warmup

from utils import MODEL_DICT, get_intent_labels, get_slot_labels, load_tokenizer
from data_loader import load_and_cache_examples
from module import Module
from main import args



class ISDFModule(Module):
    def __init__(self, args):
        super().__init__(args)
        #define model
        config, _, model  = MODEL_DICT[args.pretrained_model]
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
            self.model = model()
        #define hparams
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.n_epochs = args.n_epochs
        self.device = args.device
    
    def train_step(self, batch):
        batch = tuple(t.to(self.device) for t in batch)  # GPU or CP
        inputs = {
            "input_ids": batch[0],
            "attention_mask": batch[1],
            "intent_label_ids": batch[3],
            "slot_labels_ids": batch[4],
        }
        if "distill" not in args.pretrained_model:
            inputs["token_type_ids"] = batch[2]
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
            if "distill" not in args.pretrained_model:
                inputs["token_type_ids"] = batch[2]
            outputs = self.model(**inputs)
            tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
            loss = tmp_eval_loss.mean().item()
        # Intent prediction
        if intent_preds is None:
            intent_preds = intent_logits.detach().cpu().numpy()
            intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
        else:
            intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
            intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
        # Slot prediction
        if slot_preds is None:
            if self.args.use_crf:
                # decode() in `torchcrf` returns list with best index directly
                slot_preds = np.array(self.model.crf.decode(slot_logits))
            else:
                slot_preds = slot_logits.detach().cpu().numpy()
            slot_label_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
        else:
            if self.args.use_crf:
                slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
            slot_label_ids = inputs["slot_labels_ids"].detach().cpu().numpy()

        return {"loss": loss, "intent": intent_label_ids, "slot": slot_label_ids}

if __name__ == "__main__":
    module = ISDFModule(args)
    tokenizer = load_tokenizer(args)
    dataset = load_and_cache_examples(args, tokenizer, mode="train")
    loader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(module.model)
    print(module.train_step(batch))

    



        


        


        


