import json 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
path = "/home/tranbaochi_/Study/hust/tourxQA/data/database/test/text.json"

if __name__ == "__main__":
    context = 'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
    question = 'To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?'
    tokens = tokenizer(question,
                           context,
                           max_length=100,
                           truncation='only_second',
                           stride=50,
                           return_overflowing_tokens=True,
                           )
    print("LENGTH", [len(tokens['input_ids'][i]) for i in range(len(tokens['input_ids']))])