import json
import os
import sys
import pandas as pd

def parse_json(path):
    output = []
    with open(path, 'rb') as f:
        content = []
        for line in f.readlines():
            if len(line) > 10:
                print(line)
                a = json.loads(line)
                # print('???', a)
                break
    for line in content:
        title, content = line.split('title":')[1].split(', "content": ')
        output.append(dict([('title', title), ('content', content.encode().decode('utf-8'))]))
    return output

def parse_csv(path):
    pass


if __name__ == '__main__':
    df = pd.read_csv("/home/xps/educate/code/learn_concepts/crawl/chatbot/chatbot/spiders/test.csv")
    print(df)
    print('length', len(df))
    print('fields', list(df['title']))


            