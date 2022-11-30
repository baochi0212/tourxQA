data_dir = "PhoATIS/word-level"
def add_new_label(slot_label=["B-num_person", "I-num_person"], intent_label=[]):
    intent_write = open(data_dir + f"/intent_label.txt", "r")
    slot_write = open(data_dir + f"/slot_label.txt", "r")
    print(slot_write.readlines())
    # for slot in slot_label:
    #     slot_write.write(slot)
    # for intent in intent_label:
    #     intent_write.write(intent)




def add_new_data(seq_in, seq_out, labels):
    pass


def augment():
    pass


if __name__ == "__main__":
    add_new_label()