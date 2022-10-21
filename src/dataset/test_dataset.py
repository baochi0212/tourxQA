import os
import sys
src_dir = os.environ['src']
sys.path.append(src_dir)
import torch 
from torch.utils import data


from utils.preprocess import preprocess_fn

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'


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

        self.data = open((data_path + '/seq.in'), 'r').readlines()
        self.intent_label = [i.strip() for i in open((data_path + '/label'), 'r').readlines()]
        self.pos_label = open((data_path + '/seq.out'), 'r').readlines()
        count = 0
        self.intent_dict = {}
        for label in self.intent_label:
            if "#" not in label:
                self.intent_dict[label] = count 
                count += 1 

        self.pos_dict = dict([(k.strip(), i) for i, k in enumerate(open(pos_path, 'r').readlines())])
        self.MAX_LENGTH = MAX_LENGTH
        self.n_intent = len(self.intent_dict.keys())#UNK
        self.n_pos = len(self.pos_dict.keys())#PAD UNK
    def __getitem__(self, idx):
        '''
        pad and truncate tokens, pos_label and get the intent_label
        '''
        tokens = preprocess_fn(self.data[idx], max_length=self.MAX_LENGTH)
        intent_label = torch.tensor([1 if label in self.intent_label[idx].split("#") else 0 for label in self.intent_dict.keys()], dtype=torch.long)
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
if __name__ == '__main__':
    '''
    -check the dict label carefully
    -check the padding and truncation 
    -check the masking
    '''
    dataset = IntentPOSDataset(raw_dir, mode='train')
    dev_dataset = IntentPOSDataset(raw_dir, mode='test')
    dataloader = data.DataLoader(dataset, batch_size=32, shuffle=True)
    # # print("test: ", dataset.intent_dict, dataset.pos_dict)
    # # print("sample: ", dataset[0][0].shape, dataset[0][1])
    sample = next(iter(dataloader))
    print("sample: ", sample[0].shape, sample[2].shape)
    print("NUM: ", dataset.n_intent, dataset.n_pos, dataset.intent_dict)
    for i in range(len(dev_dataset)):
        a = dataset[i]