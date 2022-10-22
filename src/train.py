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

from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from dataset.test_dataset import IntentPOSDataset
from models.modules import IntentPOSModule, CustomConfig
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'

# Specify loss function
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def CE_loss_fn(pred, label):
    loss = nn.CrossEntropyLoss(reduction='none')(pred, label)
    loss = torch.where(label != 0, loss, torch.tensor([0.]).to(device))
    loss = loss.mean()
    return loss

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def train(model, optimizer, scheduler, train_dataloader, total_steps, epochs, val_dataloader=None, evaluation=False, overfit_batch=False):
    """
    -Set the val to None: if don't wanna keep track of validation.
    -Overfit one batch: for check sanity of model
    """
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
            intent_logits, pos_logits = model(b_input_ids, b_attn_mask)
    

            # Compute loss and accumulate the loss values
            loss_1 = nn.BCEWithLogitsLoss()(intent_logits, b_intent_labels)
            loss_2 = CE_loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), b_pos_labels.view(-1))
    
            batch_loss_1 += loss_1.item()
            batch_loss_2 += loss_2.item()
            total_loss_1 += loss_1.item()
            total_loss_2 += loss_2.item()

            # Perform a backward pass to calculate gradients
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
            val_loss_1, val_loss_2, val_accuracy_1, val_accuracy_2 = evaluate(model, val_dataloader)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss_1:^12.6f}, {avg_train_loss_2:^12.6f} | {val_loss_1:^10.6f}, {val_loss_2:^10.6f} | {val_accuracy_1:^9.2f}, {val_accuracy_2:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader, print_fn=False):
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

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_intent_labels, b_pos_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            intent_logits, pos_logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss_1 = nn.BCEWithLogitsLoss()(intent_logits, b_intent_labels)
        loss_2 = CE_loss_fn(pos_logits.view(-1, pos_logits.shape[-1]), b_pos_labels.view(-1))

        val_loss_1.append(loss_1.item())
        val_loss_2.append(loss_2.item())

        # Get the predictions
        intent_preds, pos_preds = intent_logits.view(-1) > 0,  torch.argmax(pos_logits, dim=-1).view(-1)
        # prob = nn.functional.softmax(intent_logits, dim=1)
  

        # Calculate the accuracy rate
        if print_fn:
          print("INTENT preds", intent_preds.view(-1))
          print("INTENT LABELS", b_intent_labels.view(-1))
        accuracy = (intent_preds == b_intent_labels.view(-1)).cpu().numpy().mean() * 100
        # print("intent accuracy: ", accuracy)
        val_accuracy_1.append(accuracy)
        accuracy = (pos_preds == b_pos_labels.view(-1)).cpu().numpy().mean() * 100
        val_accuracy_2.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss_1 = np.mean(val_loss_1)
    val_loss_2 = np.mean(val_loss_2)
    val_accuracy_1 = np.mean(val_accuracy_1)
    val_accuracy_2 = np.mean(val_accuracy_2)
    return val_loss_1, val_loss_2, val_accuracy_1, val_accuracy_2

if __name__ == '__main__':
    dataset = IntentPOSDataset(raw_dir, MAX_LENGTH=30)
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    config = CustomConfig(n_pos=dataset.n_pos, n_intent=dataset.n_intent)
    sample = next(iter(dataloader))
    print("SAMPLE", sample[0][0].shape)

    '''
    define train process
    '''
    train_dataset = IntentPOSDataset(raw_dir, mode='train', MAX_LENGTH=30)
    val_dataset = IntentPOSDataset(raw_dir, mode='dev', MAX_LENGTH=30)
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)
    net = IntentPOSModule(config)
    optimizer = AdamW(net.parameters(), lr=5e-4)
    epochs = 9
    total_steps = len(train_dataloader) * epochs
 
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    accelerator = Accelerator()
    net, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
                            net, optimizer, train_dataloader, val_dataloader
                            )
    train(net, optimizer, scheduler, train_dataloader, total_steps, epochs, val_dataloader=val_dataloader, evaluation=True, overfit_batch=False)
    # print(evaluate(net, val_dataloader, print_fn=True))