import pandas as pd

with open('label1.txt', 'r', encoding='utf-8') as file1:
    texts_label1 = file1.readlines()

with open('label0.txt', 'r', encoding='utf-8') as file0:
    texts_label0 = file0.readlines()

texts_label1 = [text.strip() for text in texts_label1]
texts_label0 = [text.strip() for text in texts_label0]

data = pd.DataFrame({
    'text': texts_label1 + texts_label0,
    'label': [1] * len(texts_label1) + [0] * len(texts_label0)
})

data.to_csv('dataset.csv', index=False, encoding='utf-8')
