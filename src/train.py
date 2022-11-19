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
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, PreTrainedTokenizerFast, AdamW, get_linear_schedule_with_warmup, RobertaConfig
from dataset.test_dataset import IntentPOSDataset, QADataset
from models.modules import IntentPOSModule, CRFPOS, CustomConfig, QAModule
from utils.preprocess import get_label
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=9, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--learning_rate', default=3e-5, type=float, help="Learning rate")
parser.add_argument('--pretrained_model', default='NlpHUST/bert-base-vn', type=str)
parser.add_argument('--pretrained_input', default=768, type=int)
parser.add_argument('--max_length', default=500, type=int)
parser.add_argument('--compare', action='store_true', default=False)
parser.add_argument('--fast', action='store_true', default=False, help='Using fast AutoQA')
parser.add_argument('--task', choices=['IDSF', 'VI_QUAD', 'SQUAD'], default='VI_QUAD')
parser.add_argument('--device', default='cuda', type=str)
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
squad_processed = data_dir + '/data/processed/SQUAD'
tokenizer_checkpoint = 'NlpHUST/bert-base-vn'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

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
def metrics(start, end, l_start, l_end, metrics='acc', test=False, training=True):
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
            elif not training:
                print("MISTAKES", start_end, l_start_end)
        #         if count % 10 == 0:
        #             # print("SAMPLE", start_end, l_start_end)
        # # print("COUNT:", count)
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
        # else:
        #     print("PRED: ", start_end)
        #     print("LABEL: ", l_start_end)
    # print("COUNT", count)
    return count/batch_size

def train_QA(model, optimizer, scheduler, train_dataloader, total_steps, epochs, device, val_dataloader=None, evaluation=False, overfit_batch=False):
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
            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            # print(b_input_ids.shape, b_attn_mask.shape)
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
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
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
            # _, train_accuracy = evaluate_QA(model, train_dataloader, device=device)
            val_loss, val_accuracy = evaluate_QA(model, val_dataloader, device=device)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            # , {train_accuracy:^9.2f}
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f}| {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    print("Training complete!")

def evaluate_QA(model, val_dataloader, device, tokenizer=tokenizer, print_fn=False, test=False, pipeline=False, training=True):
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
      #if using the pipeline API of huggingface
      if not pipeline:
          for batch in val_dataloader:
                # print(batch)
                # Load batch to GPU
                b_input_ids, b_attn_mask, b_start, b_end = tuple(t.to(device) for t in batch)
                if len(b_start.shape) == 2:
                    b_start_sub, b_end_sub = b_start[:, 0], b_end[:, 0]
                else:
                    b_start_sub, b_end_sub = b_start, b_end
                if not test:
                    loss, outputs  = model(b_input_ids, b_attn_mask, b_start_sub, b_end_sub)
                    val_loss.append(loss.item())
                    count += 1 
                else:

                    outputs = model(b_input_ids, b_attn_mask)
                    #double check
                    count += 1
                start_logits = outputs['start_logits']
                end_logits  = outputs['end_logits']
                start, end = torch.argmax(start_logits, -1), torch.argmax(end_logits, -1)
                # if count % 3 == 0:
                #     print("Q and C: ", tokenizer.decode(b_input_ids[0]))
                #     print("answers:", tokenizer.decode(b_input_ids[0][start[0]:end[0]+1]))
                # in squad-v2 val is same as train (1-label) but in Vi-squad it's multi-label
                val_accuracy.append(metrics(start, end, b_start, b_end, metrics='acc', test=True, training=training))
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
    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.n_epochs
    model_checkpoint = args.pretrained_model
    lr = args.learning_rate
    max_length = args.max_length
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = RobertaConfig.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = QAModule(
        model_checkpoint,
        args=args,
        device=device,
    )

    optimizer = transformers.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model_path = './models/weights/model.pt'
    train_df = pd.read_csv(qa_processed + '/train.csv')
    val_df = pd.read_csv(qa_processed + '/dev.csv')
    test_df = pd.read_csv(qa_processed + '/test.csv')
    if args.task == 'SQUAD':
        train_df = pd.read_csv(squad_processed + '/train.csv')
        val_df = pd.read_csv(squad_processed + '/validation.csv')

    train_dataset = QADataset(train_df, tokenizer=tokenizer, mode='train', MAX_LENGTH=max_length)
    val_dataset = QADataset(val_df, tokenizer=tokenizer, mode='test', MAX_LENGTH=max_length)
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test', MAX_LENGTH=max_length)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    total_steps = len(train_loader) * epochs
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0.05*total_steps, # Default value
                                                num_training_steps=total_steps)

    train_QA(model.to(device), optimizer, scheduler, train_loader, total_steps, epochs, device=device, val_dataloader=val_loader, evaluation=True, overfit_batch=False)
    model.eval()
    torch.save(model.state_dict(), model_path)