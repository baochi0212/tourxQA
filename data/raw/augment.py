data_dir = "PhoATIS/word-level"
def add_new_label(slot_label=["B-num_person", "I-num_person"], intent_label=[]):
    with open(data_dir + f"/slot_label.txt", "a") as f:
        f.write('\n')
        for line in slot_label:
            f.write(line + '\n')

    with open(data_dir + f"/intent_label.txt", "a") as f:
        f.write('\n')
        for line in intent_label:
            f.write(line + '\n')




def add_new_data(seq_in, seq_out, labels):
    pass


def augment():
    pass


if __name__ == "__main__":
    add_new_label()