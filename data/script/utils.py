import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)

def concat_train(path='/home/xps/educate/code/hust/XQA/data/processed/QA'):
    train_df = pd.read_csv(path + '/train.csv')
    val_df = pd.read_csv(path + '/dev.csv')
    pd.concat([train_df, val_df]).to_csv(path + '/concat.csv')


if __name__ == '__main__':
    args = parser.parse_args()
    concat_train(args.path)