import os

from underthesea import word_tokenize
from transformers import AutoTokenizer

data_dir = os.environ['dir']
raw_dir = data_dir + '/data/raw/PhoATIS'
processed_dir = data_dir + '/data/processed/PhoATIS'
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
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



if __name__ == '__main__':
    text = "Tôi có một khách sạn ven biển a a a a a a a a a a a a a a a a a a a a a a a a a a a a a"
    print("join", preprocess_fn(text))
    output = query_stat(['train', 'dev', 'test'])
    # print("unique: ", len(output[0]), len(output[1]), output[2], len(output[3]), len(output[4]))
    print("special: ", output[-1])
    # print(tracking_frequency("I-aircraft_code", type='pos'))