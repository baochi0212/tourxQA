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

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/ta/processed/PhoATIS'
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
tokenizer_checkpoint = 'NlpHUST/bert-base-vn'


if __name__ == '__main__':
    batch_size = 32
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_checkpoint = 'NlpHUST/bert-base-vn'
    model = QAModule(model_checkpoint=model_checkpoint, device=device).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    # model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    model_path = './models/weights/model.pt'

    test_df = pd.read_csv(qa_processed + '/test.csv')
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test')
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(evaluate_QA(model.to(device), test_loader, test=True))


