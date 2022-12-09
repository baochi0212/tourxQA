import logging
import os
import random
import re

import numpy as np
import torch
from model.IDSF_modules import JointPhoBERT, JointXLMR, JointLSTM, JointGRU, JointDistillBERT
from model.QA_modules import QARoberta
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    AutoConfig,
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    AutoModel,
    DistilBertConfig
)
from collections import Counter

#MAPPING
MODEL_DICT = {
    "phobert": (AutoConfig, AutoTokenizer, JointPhoBERT),
    "lstm": (AutoConfig, AutoTokenizer, JointLSTM),
    "gru": (AutoConfig, AutoTokenizer, JointGRU), 
    "distill-bert": (DistilBertConfig, AutoTokenizer, JointDistillBERT),
}
QA_DICT = {
    "xlm-roberta-base": (AutoConfig, AutoTokenizer, QARoberta),
    "xlm-roberta-large": (XLMRobertaConfig, XLMRobertaTokenizer, QARoberta),
    "electra": (AutoConfig, AutoTokenizer, QARoberta),
    "phobert": (AutoConfig, AutoTokenizer, JointPhoBERT),
    "lstm": (AutoConfig, AutoTokenizer, JointLSTM),
    "gru": (AutoConfig, AutoTokenizer, JointGRU),
    
}

# MODEL_PATH_MAP = {
#     "xlm-base": "xlm-roberta-base",
#     "xlm-large": "xlm-roberta-large",
#     "phobert": "vinai/phobert-base",

# }

#IDSF utils
def get_intent_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.idsf_data_dir, args.intent_label_file), "r", encoding="utf-8")
    ]


def get_slot_labels(args):
    return [
        label.strip()
        for label in open(os.path.join(args.idsf_data_dir, args.slot_label_file), "r", encoding="utf-8")
    ]


#loading and initialize
def load_tokenizer(args):
    return AutoTokenizer.from_pretrained(args.pretrained_model)


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

#metrics
def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    mean_intent_slot = (intent_result["intent_acc"] + slot_result["slot_f1"]) / 2

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)
    results["mean_intent_slot"] = mean_intent_slot

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)

    slot_result = []
    for pred, label in zip(preds, labels):
        assert len(pred) == len(label)
        one_sent_result = True
        for p, l in zip(pred, label):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)
    return {
        "slot_accuracy": slot_result.mean(),
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds),
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {"intent_acc": acc}


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), "r", encoding="utf-8")]

def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = intent_preds == intent_labels

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    semantic_acc = np.multiply(intent_result, slot_result).mean()
    return {"semantic_frame_acc": semantic_acc}
def convert_tokens_to_labels(word, label_dict):
    if word in label_dict.keys():
        return label_dict[word]
    else:
        return label_dict['UNK']

#GNN utils
def to_graph_format(files):
    '''
    - get label dict
    - read train, test, val
    '''

    path = './PhoATIS/word-level'

    label_dict = {}
    count = 0
    with open(f'{path}/intent_label.txt', 'r') as f:
        for label in f.readlines():
            label = label.strip()
            if label not in label_dict:
                label_dict[label] = count 
                count += 1

    with open(path + '/phoatis_text.txt', 'w') as f1:
        with open(path + '/phoatis_intent.txt', 'w') as f2: 
            for file in files:
                idx = 0 #for indexing in the label .txt file
                for line in open(f'{path}/{file}/seq.in', 'r').readlines():
                    f1.write(line)
                for line in open(f'{path}/{file}/label', 'r').readlines():
                    line = str(idx) + '\t' + file + '\t' + str(convert_tokens_to_labels(line.strip(), label_dict)) + '\n'
                    idx += 1 
                    f2.write(line)

#QA utils
def compare_text(pred, trues):
    '''
    check if additional non-alpha index exists
    '''
    match = False
    for true in trues:
        if pred in true:
            temp = len(true.replace(pred, ''))
            for i in true.replace(pred, ''):
                if not i.isalpha():
                    temp -= 1 
            match = temp == 0
    return match 


def QA_metrics(start, end, start_idx, end_idx, input_ids, tokenizer):
    def compute_f1(prediction, truth):
        pred_tokens = prediction.split()
        truth_tokens = truth.split()
        
        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0
        
        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)
        
        return 2 * (prec * rec) / (prec + rec)
    '''
    EM and F1 score for text output
    start = b x n
    '''
    EM_1 = 0
    F1_1 = 0
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
            F1 += 1
            EM_1 += 1 
            F1_1 += 1 
            continue
        if compare_text(pred, trues):
            EM_1 += 1
            F1_1 += 1 

        # else:
        #     print("PREDICTION:", pred)
        #     print("GROUND TRUTH:", trues)
        #     print("CONTEXT:", tokenizer.decode(input_ids[i]))
        #F1 score
        F1_score = []
        for true in trues:
            F1_score.append(compute_f1(pred, true))
            # sum = 0
            
            # text = pred if len(pred.split()) < len(true.split()) else true
            # for i in range(len(text.split())):
            #     if pred.split()[i] in true.split():
            #         sum += 1
            # if len(pred.split()) == 0 or len(true.split()) == 0:
            #     F1_score.append(int(pred == true))
            #     continue
            # precision = sum/len(pred.split())
            # recall = sum/len(true.split())
            # if precision == 0 or recall == 0:
            #     F1_score.append(0)
            #     continue


            # F1_score.append(2/(1/precision + 1/recall))
        
        F1 += max(F1_score)
        if not compare_text(pred, trues):
            F1_1 += max(F1_score)
    return EM/start.shape[0], F1/start.shape[0], EM_1/start.shape[0], F1_1/start.shape[0]
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

def get_label(input, text, start):

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
    to_graph_format(['train', 'dev', 'test'])

        
    
