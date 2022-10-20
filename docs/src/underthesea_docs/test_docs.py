import underthesea
from underthesea import sent_tokenize, word_tokenize
from transformers import AutoModel, AutoTokenizer
#sentence segmentation
text = 'Taylor cho biết lúc đầu cô cảm thấy ngại với cô bạn thân Amanda nhưng rồi mọi thứ trôi qua nhanh chóng. Amanda cũng thoải mái với mối quan hệ này.'
print(sent_tokenize(text))
#word segmentation
print(word_tokenize(text))
#phobert test and pre-installed :>
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base")
vocab = tokenizer.get_vocab()