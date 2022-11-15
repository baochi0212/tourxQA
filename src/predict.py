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
from train import evaluate_QA
import argparse

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
tokenizer_checkpoint = 'NlpHUST/bert-base-vn'


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=9, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--learning_rate', default=3e-5, type=float, help="Learning rate")
parser.add_argument('--pretrained_model', default='NlpHUST/bert-base-vn', type=str)
parser.add_argument('--pretrained_input', default=768, type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    batch_size = args.batch_size

    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = args.pretrained_model
    model = QAModule(model_checkpoint=model_checkpoint, device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    model_path = './models/weights/model.pt'

    test_df = pd.read_csv(qa_processed + '/test.csv')
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test')
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(evaluate_QA(model.to(device), test_loader, device=device, test=True))


