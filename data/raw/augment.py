import random
from underthesea import word_tokenize



dataset = ['train', 'dev']
data_dir = "/home/xps/educate/code/hust/XQA/data/raw/PhoATIS/word-level"
def add_new_label(slot_label=["B-num_person", "I-num_person"], intent_label=[]):
    with open(data_dir + f"/slot_label.txt", "a") as f:
        temp_slot = [line.strip() for line in open(data_dir + f"/slot_label.txt", "r").readlines()]
        f.write('\n')
        for slot in slot_label:
            if slot not in temp_slot:
                f.write(slot + '\n')

    with open(data_dir + f"/intent_label.txt", "a") as f:
        temp_intent = [line.strip() for line in open(data_dir + f"/intent_label.txt", "r").readlines()] 
        f.write('\n')
        for intent in intent_label:
            if intent not in temp_intent:
                f.write(intent + '\n')

def get_rand(value_list):
    i = random.randint(0, len(value_list)-1)
    return value_list[i]


def add_new_data(intents=['airfare', 'flight'], num_samples=100):
    #add new terms to filtering sentences
    def condition(i):

        if labels[i] in intents:
            return True
    def generate(words={'beginning': [['cho', 'cho nhóm'], 'O'], 'numbers': [[random.randint(0, 100)], 'B-num_person'], 'humans': [['người', 'người lớn', 'trẻ em'], 'I-num_person']}):
        word = ''
        out = ''
        for key in words.keys():

            word += ' ' + str(get_rand(words[key][0]))
            out += ' ' + words[key][1]
        return word_tokenize(word.strip(), format='text'), out.strip()
            

    for mode in dataset:
        path_in = f"/{data_dir}/{mode}/seq.in"
        path_out = f"{data_dir}/{mode}/seq.out"
        path_label = f"{data_dir}/{mode}/label"
        ins = [line.strip().split() for line in open(path_in, 'r').readlines()]
        outs = [line.strip().split() for line in open(path_out, 'r').readlines()]
        labels = [line.strip().split()[0] for line in open(path_label, 'r').readlines()]

        
        with open(path_in, 'a') as f_in:
            with open(path_out, 'a') as f_out:
                while num_samples > 0:
                    for i in range(len(labels)):
                        if condition(i):

                            if outs[-1] != 'O':
                                word, out = generate()
                                
                                f_in.write(' '.join(ins[i]) + ' ' + word + '\n')
                                f_out.write(' '.join(outs[i]) +  ' ' + out + '\n')

                                num_samples -= 1 

                            if outs[-1] == 'O' and outs[-2] != 'O':
                                word, out = generate()
                                f_in.write(' '.join(ins[i][:-1]) + ' ' + word + ' '  + ins[i][-1] + '\n')
                                f_out.write(' '.join(outs[i][:-1]) + ' ' + out + ' ' +  outs[i][-1] + '\n')
                                num_samples -= 1 
                num_samples = 100
                        

                            


                




    


def augment():
    pass


if __name__ == "__main__":
    add_new_label()
    add_new_data()
