import os
import sys
#add source directory 
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch 
from torch.utils import data
import pandas as pd

from utils.preprocess import preprocess_fn
from utils.preprocess import get_label, string2list

from transformers import AutoTokenizer

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
qa_processed = data_dir + '/data/processed/QA'
tokenizer = AutoTokenizer.from_pretrained('deepset/roberta-base-squad2')

class IntentPOSDataset(data.Dataset):
    '''
    - for: training the Intent cls and POS tag 
    - input: level: 'word' or 'syllable', with seq_in, seq_out (POS), label (intent), and 2 pre-defined label files 
    - tokenizer:
    - multi-label intent and multi-class POS :> 
    '''
    def __init__(self, data_dir, mode='train', level='word', MAX_LENGTH=10):
        path = data_dir + f'/{level}-level'
        data_path = path + f'/{mode}'
        intent_path, pos_path = path + '/intent_label.txt', path + '/slot_label.txt'

        self.data = open(data_path + '/seq.in', 'r').readlines()
        self.intent_label = [i.strip() for i in open((data_path + '/label'), 'r').readlines()]
        self.pos_label = open((data_path + '/seq.out'), 'r').readlines()
        count = 0
        self.intent_dict = {}
        for label in open(intent_path, 'r').readlines():
            if "#" not in label:
                self.intent_dict[label.strip()] = count 
                count += 1 

        self.pos_dict = dict([(k.strip(), i) for i, k in enumerate(open(pos_path, 'r').readlines())])
        self.MAX_LENGTH = MAX_LENGTH
        self.n_intent = len(self.intent_dict.keys())#UNK
        self.n_pos = len(self.pos_dict.keys())#PAD UNK
        self.n_intent = len(self.intent_dict.keys())
        self.n_pos = len(self.pos_dict.keys())
    def __getitem__(self, idx):
        '''
        pad and truncate tokens, pos_label and get the intent_label
        '''
        tokens = preprocess_fn(self.data[idx], max_length=self.MAX_LENGTH)
        intent_label = torch.tensor([1 if label in self.intent_label[idx].split("#") else 0 for label in self.intent_dict.keys()], dtype=torch.float)
        pos_label = torch.tensor([self.pos_dict[self.pos_label[idx].strip().split(' ')[i]] for i in range(len(self.pos_label[idx].strip().split(' ')))], dtype=torch.long)
        if 2 in tokens: #[SEP] not be truncated
            pos_label = torch.cat([torch.tensor([0]), pos_label, torch.tensor([0])], dim=-1) #cls and sep token will have pos-tag 0
        else:
            pos_label = torch.cat([torch.tensor([0]), pos_label], dim=-1)
        #pad to max or truncate pos_label
        if pos_label.shape[-1] < self.MAX_LENGTH:
            pos_label = torch.cat([pos_label, torch.zeros(self.MAX_LENGTH - pos_label.shape[-1], dtype=torch.long)], dim=-1)
        if pos_label.shape[-1] > self.MAX_LENGTH:
            pos_label = pos_label[:self.MAX_LENGTH]
        return tokens['input_ids'].view(-1), tokens['attention_mask'].view(-1), intent_label, pos_label
    def __len__(self):
        return len(self.data)

class QADataset(data.Dataset):
    def __init__(self, df, tokenizer, MAX_LENGTH=222, mode='train', pipeline=False):
        self.df = df
        self.MAX_LENGTH = MAX_LENGTH
        self.mode = mode
        self.tokenizer = tokenizer
        self.pipeline = pipeline
        self.max_answer_length = 4 

    def __getitem__(self, idx):
        '''
        TEST SET: more answer options and will compare with the PIPELINE api
        '''
        item = self.df.iloc[idx]
        q, c, answer_start, text = item['question'], item['context'], string2list(item['start'], type='int'), string2list(item['text'])
        # print(len(answer_start), len(text))
        # if len(text) > 4:
        #     print(text)
        input = self.tokenizer(q.strip(), c, return_tensors='pt',
            max_length=self.MAX_LENGTH,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",)
        if self.mode != 'test':
            if len(text) == 0:
                start, end = torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)
            else:
                start, end = get_label(input, text[0], answer_start[0])
                if start > self.MAX_LENGTH or self.MAX_LENGTH < end: 
                    start, end = torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)

        else:
            if len(text) == 0:
                start_list, end_list = [0], [0]
            else:
                start_list, end_list = [], []
                for i in range(len(text)):
                    start, end = get_label(input, text[i], answer_start[i])
                    if start > self.MAX_LENGTH or self.MAX_LENGTH < end: 
                        start, end = torch.tensor(0, dtype=torch.long), torch.tensor(0, dtype=torch.long)
                    start_list.append(start.item())
                    end_list.append(end.item())
            # max length of test answers
            while len(start_list) < self.max_answer_length:
                start_list.append(-1)
                end_list.append(-1)
            start, end = torch.tensor(start_list), torch.tensor(end_list)

            if self.pipeline:
                return input['input_ids'][0], input['attention_mask'][0], start, end, q, c, input['offset_mapping'][0]
        return input['input_ids'][0], input['attention_mask'][0], start, end
    def __len__(self):
        return len(self.df)

if __name__ == '__main__':
    '''
    -check the dict label carefully
    -check the padding and truncation 
    -check the masking
    -check the label set carefully!!!
    '''
    # dataset = IntentPOSDataset(raw_dir, mode='train')
    # dev_dataset = IntentPOSDataset(raw_dir, mode='dev')
    # dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # # # print("test: ", dataset.intent_dict, dataset.pos_dict)
    # # # print("sample: ", dataset[0][0].shape, dataset[0][1])
    # sample = next(iter(dataloader))
    # print("sample: ", sample[2], sample[2].shape, dataset.intent_dict)
    # # print("dict: ", dataset.intent_label, len(dataset.intent_label))
    # # for i in range(len(dev_dataset)):
    # #     a = dataset[i]

    test_df = pd.read_csv(qa_processed + '/train.csv')
    test_dataset = QADataset(test_df, tokenizer=tokenizer, mode='test')
    test_loader = data.DataLoader(test_dataset, batch_size=16)
    for i in test_loader:
        print(i[-1].shape)