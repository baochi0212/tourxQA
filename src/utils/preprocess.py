from underthesea import word_tokenize
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def preprocess_fn(text, max_length=10):
    def join_fn(word):
        word = '_'.join([w for w in word.split(' ')])
        return word
    text = ' '.join([join_fn(word) for word in word_tokenize(text)])
    text = tokenizer(text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')
    return text

if __name__ == '__main__':
    text = "Tôi có một khách sạn ven biển"
    print("join", preprocess_fn(text), tokenizer.decode(preprocess_fn(text)['input_ids'].view(-1)))
