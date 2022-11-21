import os
import json
import numpy as np
import re

import torch
from underthesea import word_tokenize
from transformers import AutoTokenizer
from glob import glob
import pandas as pd
# from datasets import load_dataset
#Intent&SLOT
data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
#QA
qa_dir = data_dir + '/data/raw/ViSquadv1.1'
qa_processed = data_dir + '/data/processed/QA'
squad_processed = data_dir + '/data/processed/SQUAD'
'''
- preprocess text
- query statistics of dataset 
'''
def preprocess_fn(text, max_length=5):
    def join_fn(word):
        word = '_'.join([w for w in word.split(' ')])
        return word
    text = ' '.join([join_fn(word) for word in word_tokenize(text)])
    text = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    return text
def tracking_frequency(label, modes=['test'], type='intent'):
    '''
    track the frequency of label for intent and pos in modes (train, dev, test)
    '''
    count = 0
    for mode in modes:
        if type == 'intent':
            for l in open(raw_dir + f'/word-level/{mode}/label').readlines():
                l = l.strip()
                if label == l:
                    count += 1 

        else:
            for labels in open(raw_dir + f'/word-level/{mode}/seq.out').readlines():
                for l in labels.split(' '):
                    l = l.strip()
                    if label == l:
                        count += 1 

    return count 


        

    
def query_stat(files=['train', 'dev', 'test']):
    '''
    - query the num of unique pos and unique intent
    - query the num samples
    - compare with the provided labels
    - get the test_special
    '''
    og_intent, og_pos = [], []
    intent_label = []
    pos_label = []
    test_special = {'intent': [], 'pos': []}
    num_samples = dict([(file, 0) for file in files])
    for file in files:
        for label in open(raw_dir + f'/word-level/{file}/label').readlines():
            num_samples[file] += 1 
            label = label.strip()
            if label not in intent_label:
                if file == 'test':
                    test_special['intent'].append([label, tracking_frequency(label, type='intent')])
                intent_label.append(label)
        for label in open(raw_dir + f'/word-level/{file}/seq.out').readlines():
            for pos in label.split(' '):
                # pos = pos.split('-')[-1].strip() #if not differentiate B and I
                pos = pos.strip()
                if pos not in pos_label:
                    pos_label.append(pos)
                    if file == 'test':
                        test_special['pos'].append([pos, tracking_frequency(pos, type='pos')])
    for file in ['intent_label', 'slot_label']:
        for label in open(raw_dir + f'/word-level/{file}.txt'):
            # label = label.split('-')[-1].strip()
            label = label.strip()
            if file == 'intent_label':
                if label not in og_intent:
                    og_intent.append(label)
            else: 
                if label not in og_pos:
                    og_pos.append(label)
                

    return intent_label, pos_label, num_samples, og_intent, og_pos, test_special

def get_corpus(path):
    for file in glob(path + "/*"):

        data_dict = dict([(i, []) for i in ['title', 'question', 'context', 'start', 'text']])
        data = json.load(open(file, 'rb'))['data']
        for i in range(len(data)):
            #topic i
            for j in range(len(data[i]['paragraphs'])):
                #para j (with one context and many QAs)
                title = data[i]['title']
                sample = data[i]['paragraphs'][j]
                c = sample['context']
                for k in range(len(sample['qas'])):
                    #QA k (with one question and some answers)
                    qa = sample['qas'][k]
                    q = qa['question'] 
                    a = [int(item['answer_start']) for item in qa['answers']]
                    text = ''
                    for item in qa['answers']:
                        text += '@@@' + item['text']
                    #add to dataframe
                    data_dict['title'].append(title)
                    data_dict['question'].append(q)
                    data_dict['start'].append(a)
                    data_dict['text'].append(text)
                    data_dict['context'].append(c)

        df = pd.DataFrame.from_dict(data_dict)
        for i in ['train', 'dev', 'test']:
            if i in os.path.basename(file):
                basename = i
        df.to_csv(qa_processed + f'/{basename}.csv', index=False)

def offset2length(offset_map):
    word_lengths = []
    length = 1
    offset_map = [i for i in offset_map if i.sum() != 0]
    print(offset_map)
    
    for idx in range(1, len(offset_map)-1):
        if offset_map[idx][1] == offset_map[idx+1][0]:
            length += 1
        else:
            word_lengths.append(length)
            length = 1
    return word_lengths
def QA_metrics(start, end, start_idx, end_idx, input_ids, tokenizer):
    '''
    EM and F1 score for text output
    start = b x n
    '''
    EM = 0
    F1 = 0
    for i in range(start.shape[0]):
        pred = tokenizer.decode(input_ids[i][start[i]:end[i]+1])
        trues = []
        for j in range(len(start_idx[i])):
            trues.append(tokenizer.decode(input_ids[i][start_idx[i][j]:end_idx[i][j]+1]))
        #exact match
        if pred in trues:
            EM += 1 
        sum = 0
        #F1 score
        F1_score = []
        for true in trues:
            
            text = pred if len(pred.split()) < len(true.split()) else true
            for i in range(len(text.split())):
                if pred.split()[i] == true.split()[i]:
                    sum += 1 
            if len(pred.split()) == 0 or len(true.split()) == 0:
                F1_score.append(int(pred == true))
                continue
            precision = sum/len(pred.split())
            recall = sum/len(true.split())
            if precision == 0 or recall == 0:
                F1_score.append(0)
                continue
            
            F1_score.append(2/(1/precision + 1/recall))
        print(max(F1_score), pred, trues) if max(F1_score) > 0.25
        F1 += max(F1_score)
    return EM/start.shape[0], F1/start.shape[0]
    
def align_matrix():
    pass

def char2idx(start, end, context):
    subtext = context[start:end+1]
    for i in range(len(context.split())):
        if context.split()[i] == subtext.split()[0] and context.split()[i+len(subtext.split())-1] == subtext.split()[-1]:
            return i, i+len(subtext.split()) - 1
    
def get_label(input, text, start, reverse=False, max_length=300, context=None, question=None):
    '''
    we can make use of original sequence or sub_word sequence (tokenized for labelling)
    - non-reverse: use the mapping to map start and end idx
    - reverse: og start and end idx, and word lengths for mapping, use context for get start and end
    '''

        



        

        
        



    sequence_ids = input.sequence_ids()
    idx = 0
    while sequence_ids[idx] != 1:
            idx += 1
    context_start = idx
    while sequence_ids[idx] == 1 and idx < len(sequence_ids) - 1:
            idx += 1
    context_end = idx - 1
    offset = input['offset_mapping']
    

    start_positions, end_positions = 0, 0
    start_char = start
    end_char = start + len(text)
    offset = input['offset_mapping'][0]
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_positions = 0
        end_positions = 0
    else:
        # Otherwise it's the start and end token positions
        idx = context_start
        while idx <= context_end and offset[idx][0] <= start_char:
            idx += 1
        start_positions = idx - 1
        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_positions = idx + 1
        if reverse:
            start_positions, end_positions = char2idx(start_positions, end_positions, question + " " + context)
            offset_mapping = input.offset_mapping[0]
            word_lengths = offset2length(offset_mapping)

            while len(word_lengths) < max_length:
                word_lengths.append(1)

            return torch.tensor(start_positions, dtype=torch.long), torch.tensor(end_positions, dtype=torch.long), torch.tensor(word_lengths, dtype=torch.long)

 
    return torch.tensor(start_positions, dtype=torch.long), torch.tensor(end_positions, dtype=torch.long)

def string2list(text, type='str'):
    '''Convert a list_string to original list''' 
    ''' 1-D arr'''
    if type == 'int':
        text = list([int(i) for i in re.sub('[\[\]]', '' , text).split(',')])
    if type == 'str':
        temp = []
        for i in text.split('@@@'):
            if i != '':
                temp.append(i)
        text = temp


    return text

def get_corpus_squad():

  dataset = load_dataset("squad")
  for file in ['train', 'validation']:
    data_dict = dict([(i, []) for i in ['title', 'question', 'context', 'start', 'text']])
    data = dataset[file]
    for i in range(len(data)):
            sample = data[i]
            c = sample['context']
            qa = sample['answers']
            q = sample['question'] 
            a = [int(item) for item in qa['answer_start']]
            text = ''
            for item in qa['text']:
                text += '@@@' + item
            #add to dataframe
            data_dict['title'].append(sample['title'])
            data_dict['question'].append(q)
            data_dict['start'].append(a)
            data_dict['text'].append(text)
            data_dict['context'].append(c)
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(squad_processed + f'/content/{file}.csv', index=False)
            


            

    
    



if __name__ == '__main__':
    get_corpus_squad()
    # text = str(['các vùng người Đức', '@', 'các vùng người Đức', '@', '"các vùng người Đức"', '@', 'các vùng người Đức', '@'])
    # # print(text)
    # print(string2list(text))
