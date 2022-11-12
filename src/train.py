import random
import time
import numpy as np
from tqdm.auto import tqdm
import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch
import torch.nn as nn
import torchvision
from torch.utils import data
from accelerate import Accelerator
from torchcrf import CRF 
import pandas as pd 

import transformers
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset.test_dataset import IntentPOSDataset, QADataset
from models.modules import IntentPOSModule, CRFPOS, CustomConfig, QAModule
from utils.preprocess import get_label

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
tokenizer_checkpoint = 'nguyenvulebinh/vi-mrc-large'

# Specify loss function
def CE_loss_fn(pred, label):
    loss = nn.CrossEntropyLoss(reduction='none')(pred, label)
    loss = torch.where(label != 0, loss, torch.tensor([0.]).to(device))
    loss = loss.mean()
    return loss
def BCE_loss_fn(pred, label):
    # batch x logits bce loss
    loss = nn.BCELoss(reduction='none')(pred, label)
    # loss = torch.where(label != 0, loss, loss*0.5) #put weight
    return loss.mean()

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, scheduler, train_dataloader, total_steps, epochs, val_dataloader=None, evaluation=False, overfit_batch=False, crf=False):
    """
    -Set the val to None: if don't wanna keep track of validation.
    -Overfit one batch: for check sanity of model
    -init model before training
    """
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)
    # model.apply(init_weights)
    #for overfit
    fixed_batch = next(iter(train_dataloader))
    # Start training loop
    print("Start training...\n")
    run = tqdm(range(total_steps))
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss_1, total_loss_2, batch_loss_1, batch_loss_2, batch_counts = 0, 0, 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            '''
            NOTICE: CUSTOM HERE!!!
            START 
            '''
            if overfit_batch:
                batch = fixed_batch
            b_input_ids, b_attn_mask, b_intent_labels, b_pos_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            # print(b_input_ids.shape, b_attn_mask.shape)
            # Perform a forward pass. This will return logits.
            if not crf:
                intent_logits, pos_logits = model(b_input_ids, b_attn_mask)
    

                # Compute loss and accumulate the loss values
                loss_1 = BCE_loss_fn(intent_logits, b_intent_labels)
                loss_2 = CE_loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), b_pos_labels.view(-1))
            else:
                intent_logits, pos_logits, loss_2 = model(b_input_ids, b_pos_labels, b_attn_mask)
                loss_1 = BCE_loss_fn(intent_logits, b_intent_labels)
            batch_loss_1 += loss_1.item()
            batch_loss_2 += loss_2.item()
            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()

            # Perform a backward pass to calculate gradients
            # (loss_1 + loss_2).backward()
            (loss_1 + loss_2).backward()
            '''
            END!!!! (MODIFY THE VAL LOADER AS WELL, and maybe LOSS PRINTER)
            '''

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            # scheduler.step()
            run.update(1)


            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss_1 / batch_counts:^12.6f},{batch_loss_2 / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss_1, batch_loss_2, batch_counts = 0, 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss_1 = total_loss_1/len(train_dataloader)
        avg_train_loss_2 = total_loss_2/len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss_1, val_loss_2, val_accuracy_1, val_accuracy_2 = evaluate(model, val_dataloader, crf=crf)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss_1:^12.6f}, {avg_train_loss_2:^12.6f} | {val_loss_1:^10.6f}, {val_loss_2:^10.6f} | {val_accuracy_1:^9.2f}, {val_accuracy_2:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    print("Training complete!")


def evaluate(model, val_dataloader, print_fn=False, crf=False):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy_1 = []
    val_accuracy_2 = []
    val_loss_1 = []
    val_loss_2 = []
    temp = 0
    set_seed(3)
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_intent_labels, b_pos_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        if not crf:
            with torch.no_grad():
                intent_logits, pos_logits = model(b_input_ids, b_attn_mask)
        else:
            with torch.no_grad():
                intent_logits, pos_logits, loss_2 = model(b_input_ids, b_pos_labels, b_attn_mask)

            

        # Compute loss
        loss_1 = BCE_loss_fn(intent_logits, b_intent_labels)
        loss_2 = CE_loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), b_pos_labels.view(-1))

        val_loss_1.append(loss_1.item())
        val_loss_2.append(loss_2.item())
        if not crf:
            with torch.no_grad():
                intent_logits, pos_logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss_1 = BCE_loss_fn(intent_logits, b_intent_labels)
            loss_2 = CE_loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), b_pos_labels.view(-1))
        else:
            with torch.no_grad():
                intent_logits, pos_logits, loss_2 = model(b_input_ids, b_pos_labels, b_attn_mask)

            loss_1 = BCE_loss_fn(intent_logits, b_intent_labels)

        # Get the predictions
        intent_preds, pos_preds = intent_logits > 0.5,  torch.argmax(pos_logits, dim=-1).view(-1)
        # prob = nn.functional.softmax(intent_logits, dim=1)
  

        # Calculate the accuracy rate
        #INTENT accuracy
        accuracy = 0
        for i in range(intent_preds.shape[0]):
            # print((intent_preds[i] == b_intent_labels[i]).type(torch.float32).sum())
            # # print("Logits: ", intent_logits[i])
            if (intent_preds[i] == b_intent_labels[i]).type(torch.float32).sum() == intent_preds.shape[1]:
            #     print(intent_preds[i], b_intent_labels[i])
               
                accuracy += 1 
                # if temp == 0:
                #     print(intent_preds[i], b_intent_labels[i])
            #     break

        accuracy = accuracy/intent_preds.shape[0] * 100
        # print("intent accuracy: ", accuracy)
        val_accuracy_1.append(accuracy)
        #POS accuracy
        accuracy = 0
        count = 0
        for i in range(b_pos_labels.view(-1).shape[0]):
            if b_pos_labels.view(-1)[i] != 0:
                count += 1
                if b_pos_labels.view(-1)[i] == pos_preds[i]:
                    accuracy += 1 
        accuracy = accuracy/count * 100
        val_accuracy_2.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss_1 = np.mean(val_loss_1)
    val_loss_2 = np.mean(val_loss_2)
    val_accuracy_1 = np.mean(val_accuracy_1)
    val_accuracy_2 = np.mean(val_accuracy_2)
    return val_loss_1, val_loss_2, val_accuracy_1, val_accuracy_2
def metrics(start, end, l_start, l_end, metrics='acc', test=False):
  '''
  compare the pred with label ('l')
  '''
  if not test:
      start, end, l_start, l_end = start.view(-1), end.view(-1), l_start.view(-1), l_end.view(-1)
      count = 0
      if metrics == 'acc':
        for i in range(start.shape[0]):
          if start[i] == l_start[i] and end[i] == l_end[i]:
             count += 1 
        return count/start.shape[0]
  else:
      start, end = start.view(-1), end.view(-1)
      l_start, l_end =  l_start.view(start.shape[0], -1), l_end.view(end.shape[0], -1)
      count = 0
      if metrics == 'acc':
        for i in range(start.shape[0]):
            start_end = (start[i].item(), end[i].item())
            l_start_end  = [(m.item(), n.item()) for m, n in zip(l_start[i], l_end[i])]
            if start_end in l_start_end:
                count += 1 
        return count/start.shape[0]
def metrics_pipeline(mapping, start, end, l_start, l_end):
    '''
    convert string idx to token idx , USED IN TEST SET ONLY
    '''
    count = 0
    l_start, l_end =  l_start.view(batch_size, -1), l_end.view(batch_size, -1)
    for i in range(batch_size):
        start, end = list(i.item() for i in get_label(mapping[i], '', start, end))
        start_end = (start, end)
        l_start_end  = [(m.item(), n.item()) for m, n in zip(l_start[i], l_end[i])]
        if start_end in l_start_end:
            count += 1 
        else:
            print("PRED: ", start_end)
            print("LABEL: ", l_start_end)
    print("COUNT", count)
    return count/batch_size

def train_QA(model, optimizer, scheduler, train_dataloader, total_steps, epochs, val_dataloader=None, evaluation=False, overfit_batch=False):
    """
    -Set the val to None: if don't wanna keep track of validation.
    -Overfit one batch: for check sanity of model
    -init model before training
    """
    # def init_weights(m):
    #     if isinstance(m, nn.Linear):
    #         torch.nn.init.xavier_uniform(m.weight)
    #         m.bias.data.fill_(0.01)
    # model.apply(init_weights)
    #for overfit
    fixed_batch = next(iter(train_dataloader))
    # Start training loop
    print("Start training...\n")
    run = tqdm(range(total_steps))
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            '''
            NOTICE: CUSTOM HERE!!!
            START 
            '''
            if overfit_batch:
                batch = fixed_batch
            b_input_ids, b_attn_mask, b_end, b_start = tuple(t.to(device) for t in batch)
            print("????", b_input_ids.shape, b_attn_mask.shape, b_end.shape, b_start.shape)
            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            print(b_input_ids.shape, b_attn_mask.shape)
            # Perform a forward pass. This will return logits.
            loss, _ = model(b_input_ids, b_attn_mask, b_end, b_start)
            batch_loss += loss.item()
            total_loss += loss.item()
            # Perform a backward pass to calculate gradients
            loss.backward()
            '''
            END!!!! (MODIFY THE VAL LOADER AS WELL, and maybe LOSS PRINTER)
            '''

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            # scheduler.step()
            run.update(1)
            

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f}| {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss , batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss/len(train_dataloader)

        print("-"*70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate_QA(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f}| {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    print("Training complete!")

def evaluate_QA(model, val_dataloader, print_fn=False, test=False, pipeline=False):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    count = 0
    with torch.no_grad():
      if not pipeline:
          for batch in val_dataloader:
                # print(batch)
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_start, b_end = tuple(t.to(device) for t in batch)


    #             loss, outputs  = model(b_input_ids, b_attn_mask, b_start, b_end)
                outputs = model(b_input_ids, b_attn_mask)
                start_logits = outputs['start_logits']
                end_logits  = outputs['end_logits']
                # print("START, END")
                start, end = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)
    #             val_loss.append(loss.item())
                val_accuracy.append(metrics(start, end, b_start, b_end, metrics='acc', test=test))
          val_loss =  np.array(val_loss).mean()
          val_accuracy = np.array(val_accuracy).mean()
      else:
          for batch in val_dataloader:
                 b_input_ids, b_attn_mask, b_start, b_end, q, c, mapping = batch
                 q = list(item for item in q)
                 c = list(item for item in c)
                 QA_input = {
                            'question': q,
                            'context': c
                        }
                 res = nlp(QA_input)
                 start = [item['start'] for item in res]
                 end = [item['end'] for item in res]
                 val_accuracy.append(metrics_pipeline(mapping, start, end, b_start, b_end))
          val_accuracy = np.array(val_accuracy).mean()

                 
                 
                
          



          

    return val_loss, val_accuracy

if __name__ == '__main__':
    batch_size = 16 
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = 'vinai/phobert-base'
    model = QAModule(model_checkpoint=model_checkpoint).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    optimizer = transformers.AdamW(model.parameters(), lr=5e-5)
    epochs = 9

    train_df = pd.read_csv(qa_processed + '/train.csv')
    val_df = pd.read_csv(qa_processed + '/dev.csv')
    test_df = pd.read_csv(qa_processed + '/test.csv')
    train_dataset = QADataset(test_df, tokenizer=tokenizer, mode='train')
    val_dataset = QADataset(val_df, tokenizer=tokenizer, mode='test')
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test')
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    total_steps = len(train_loader) * epochs
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)

    train_QA(model.to(device), optimizer, scheduler, train_loader, total_steps, epochs, val_dataloader=val_loader, evaluation=True, overfit_batch=False)
    # test_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    # print(evaluate_QA(model.to(device), test_loader, test=False))
    # for i in range(len(test_loader)):
    #     print(next(iter(test_loader))[-1].shape)
    # test_df = pd.read_csv(qa_processed + '/test.csv')
    # test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test')
    # test_loader = data.DataLoader(test_dataset, batch_size=32)
    # for i in range(len(test_loader)):
    #     print(next(iter(test_loader))[-1].shape)
