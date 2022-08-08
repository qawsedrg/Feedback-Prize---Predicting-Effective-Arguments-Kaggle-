import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import numpy as np
import pandas as pd

device = torch.device('cuda:{:}'.format(0))

batch_size = 2

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
tokenizer.model_max_length = 512

model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
model.to(device)

text = pd.read_csv("../text.csv")
loader = DataLoader(Feedback_Data_unlabeled(text, tokenizer),
                    batch_size=batch_size,
                    shuffle=False,
                    pin_memory=False,
                    num_workers=0, drop_last=False)

res = []
model.eval()
for data in tqdm(loader):
    with torch.no_grad():
        composed_dict = {"input_ids": data[0].squeeze(1).to(device).long(),
                         "attention_mask": data[1].squeeze(1).to(device).long()}
        res.append(model(**composed_dict).last_hidden_state[:, 0, :].cpu().numpy())
res = np.concatenate(res, axis=0)

np.save("last_hidden_state.npy", res)
