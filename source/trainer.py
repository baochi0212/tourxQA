import logging
import numpy as np
import os
import sys
sys.path.append(os.environ['source'])
import torch
from torch.utils import data

from tqdm.auto import tqdm
from early_stopping import EarlyStopping

from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from data_loader import load_and_cache_examples
from modules.IDSF import *
from main import args

from data_loader import load_and_cache_examples
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, module):
        #argument
        self.args = args
        #module
        self.module = module
        self.model = self.module.model
        self.device = args.device
    def configure_optimizer(self, t_total):
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
        #optimizer + scheduler
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )
        if "bert" not in self.args.model_type:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            scheduler = None
        return {'optimizer': optimizer, 'scheduler': scheduler}
        

    
    def fit(self, train_dataset, val_dataset):
        #loader
        print("BACTH ")
        logger.info("--MODEL CHECKING--")
        print("MODEL: ", self.model)
        
        train_sampler = data.RandomSampler(train_dataset)
        train_dataloader = data.DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, drop_last=True)
        #total_steps
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.n_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.n_epochs
        #test the initialized model on validation set
        logger.info("-----------check init---------------")
        results = self.eval(val_dataset, mode="dev")
        print(results)
        # Prepare optimizer and schedule (linear warmup and decay)
        optim = self.configure_optimizer(t_total)
        self.optimizer, self.scheduler = optim['optimizer'], optim['scheduler']


        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", self.args.n_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = range(int(self.args.n_epochs))
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
            print("\nEpoch", _)

            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                train_step = self.module.train_step(batch)
                loss = train_step['loss']
                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()
                    if self.scheduler:
                        self.scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        print("\nTuning metrics:", self.args.tuning_metric)
                        results = self.eval(val_dataset, mode="dev")
                        early_stopping(results[self.args.tuning_metric], self.model, self.args)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            break

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                train_iterator.close()
                break


        return global_step, tr_loss / global_step
        


    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def eval(self, dataset, mode="dev"):

        eval_sampler = data.SequentialSampler(dataset)
        eval_dataloader = data.DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
  
            batch = tuple(t.to(self.device) for t in batch)
            eval_step = self.module.eval_step(batch)
            tmp_eval_loss = eval_step["loss"]
            intent_logits = eval_step["intent"]
            slot_logits = eval_step["slot"]
            inputs = eval_step["inputs"]
            batch = tuple(t.to(self.device) for t in batch)
            eval_loss += tmp_eval_loss
            nb_eval_steps += 1



            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs["intent_label_ids"].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs["intent_label_ids"].detach().cpu().numpy(), axis=0
                )

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(
                    out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0
                )

            


        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.module.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != args.ignore_index:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        if mode == "test":
            self.write_evaluation_result("eval_test_results.txt", results)
            print("TEST RESULTS: ", results)
        elif mode == "dev":
            self.write_evaluation_result("eval_dev_results.txt", results)
            print("DEV: ", results)
        
        return results

    def predict(self, test_dataset):
        self.load()
        self.eval(test_dataset, mode="test")

    def save(self):
        torch.save(self.model.state_dict(), self.args.idsf_model_dir + f"{self.args.model_type}/{self.args.n_epochs}/{self.args.learning_rate}")

    def load(self):
        self.model.load_state_dict(torch.load(self.args.idsf_model_dir + f"{self.args.model_type}/{self.args.n_epochs}/{self.args.learning_rate}"))
        
if __name__ == "__main__":
    module = ISDFModule(args)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    val_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")
    trainer = Trainer(args, module)
    trainer.fit(train_dataset, val_dataset)
    
    