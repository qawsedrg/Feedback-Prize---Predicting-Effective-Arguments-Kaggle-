import pickle

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
tokenizer.model_max_length = 512

text = pd.read_csv("text.csv")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for i, (train_idxs, test_idxs) in enumerate(
        cv.split(text, y=text[["discourse_effectiveness"]], groups=text[["discourse_type"]])):
    data = []
    for idx in tqdm(train_idxs):
        composed_text, _, composed_text_mask = tokenizer.encode_plus(text.iloc[idx]["composed_text"],
                                                                     padding='max_length', truncation=True,
                                                                     add_special_tokens=True,
                                                                     return_tensors='pt').values()
        data.append([composed_text, composed_text_mask, text.iloc[idx]["discourse_effectiveness"]])
    with open("cv{}.train.pkl".format(i), "wb") as fp:
        pickle.dump(data, fp)
    data = []
    for idx in tqdm(test_idxs):
        composed_text, _, composed_text_mask = tokenizer.encode_plus(text.iloc[idx]["composed_text"],
                                                                     padding='max_length', truncation=True,
                                                                     add_special_tokens=True,
                                                                     return_tensors='pt').values()
        data.append([composed_text, composed_text_mask, text.iloc[idx]["discourse_effectiveness"]])
    with open("cv{}.eval.pkl".format(i), "wb") as fp:
        pickle.dump(data, fp)
