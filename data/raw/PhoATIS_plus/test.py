from pprint import pprint
import os
folder = os.getcwd()
train_dir_raw = folder + "/word-level/train/seq.in"
train_dir_pos = folder + "/word-level/train/seq.out"
train_dir = folder + "/word-level/train/seq.in"
test_dir = folder + "/word-level/test/seq.in"
with open(train_dir_raw, 'r') as f:
    train_raw = f.readlines()
with open(train_dir_pos, 'r') as f:
    train_pos = f.readlines()

# with open(test_dir, 'r') as f:
#     print('LENGTH test', len(f.readlines()))
for i in range(len(train_pos)):
    if "B-cost_relative" in train_pos[i]:
        print(train_raw[i], train_pos[i])
        