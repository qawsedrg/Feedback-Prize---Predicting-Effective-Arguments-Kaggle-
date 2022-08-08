import gc
import warnings

warnings.filterwarnings("ignore")

from torch.utils.data import Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig

gc.collect()

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import torchmetrics
from utils_ import Feedback_Data2
from focalloss import FocalLoss
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
import pickle

cv_id = 4
device = torch.device('cuda:{:}'.format(0))


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)  #
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeedBackModel(nn.Module):
    def __init__(self, model_name):
        super(FeedBackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_config(self.config)
        self.drop = nn.Dropout(p=0.1)
        self.pooler = MeanPooling()
        self.fc = nn.Linear(self.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids,
                         attention_mask=attention_mask,
                         output_hidden_states=False)
        out = self.pooler(out.last_hidden_state, attention_mask)
        out = self.drop(out)
        outputs = self.fc(out)
        return outputs


batch_size = 18
warmup_ratio = 0.1
scheduler_type = 'cos'

rate_decay = 0.9
base_rate = 1e-5

weight_decay = 0.01
total_epochs = 10

criterion_train = FocalLoss(gamma=0.5)
criterion_eval = nn.CrossEntropyLoss()

train_text = pickle.load(open("cv{:}.train.pkl".format(cv_id), "rb"))
eval_text = pickle.load(open("cv{:}.eval.pkl".format(cv_id), "rb"))
unlabeled_text = pd.read_csv("unlabeled_text.csv")
print("Data successfuly loaded.")

tokenizer = AutoTokenizer.from_pretrained('deberta-base/tokenizer')
tokenizer.model_max_length = 512

model = FeedBackModel('deberta-base/config.json').to(device)
model.load_state_dict(torch.load("deberta-base/models-deberta-v3-base-deberta-v3-base_fold{:}_best.pth".format(cv_id),
                                 map_location=device))


class Feedback_Data3(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        return self.df[idx]

    def __len__(self):
        return len(self.df)


train_loader = DataLoader(Feedback_Data3(train_text),
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=0, drop_last=False)

eval_loader = DataLoader(Feedback_Data3(eval_text),
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=False,
                         num_workers=0, drop_last=False)

unlabeled_loader = DataLoader(Feedback_Data2(unlabeled_text, tokenizer),
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=False,
                              num_workers=0, drop_last=False)

num_steps = len(train_loader) * total_epochs

"""
optimizer_grouped_parameters = [\
    {'params': [p for n,p in model.model.named_parameters() if n[:14]=='encoder.layer.' and int(n[14:n.find('.',15)])== i],'lr':base_rate*(rate_decay**(23-i))} for i in range(24)
    ] + [\
    {'params': [p for n,p in model.model.named_parameters() if n[:11]=='embeddings.'],'lr':base_rate*(rate_decay**23)},
    {'params':model.fc.parameters()}
        ]
"""
# optimizer=Adam(optimizer_grouped_parameters,lr=base_rate,weight_decay = weight_decay)
optimizer = AdamW([{'params': model.fc.parameters(), 'lr': 1e-5}, {'params': model.model.parameters(), 'lr': 2e-6}],
                  weight_decay=weight_decay)

if scheduler_type == "cos":
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps * warmup_ratio),
                                                num_training_steps=num_steps)
elif scheduler_type == "linear":
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps * warmup_ratio),
                                                num_training_steps=num_steps)
# scheduler1 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, total_iters=int(num_steps*warmup_ratio))
# scheduler2 = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor = 0.001, total_iters=int(num_steps*(1-warmup_ratio)))
# warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=int(warm_up_rate*num_steps))

metric_acc_train = torchmetrics.Accuracy().cpu()
metric_loss_train = torchmetrics.MeanMetric().cpu()
metric_acc_eval = torchmetrics.Accuracy().cpu()
metric_loss_eval = torchmetrics.MeanMetric().cpu()
print("Start training.")
best_loss = 10
idx = 0
iter = 0


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='model.model.embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='model.model.embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


fgm = FGM(model)
step = 0
T = 0
TT = 500
T_interval = 100
first = True
for epoch in range(total_epochs):
    model.train()
    with tqdm(train_loader) as t:
        for data in t:
            iter += 1
            composed_dict = {"input_ids": data[0].squeeze(1).to(device).long(),
                             "attention_mask": data[1].squeeze(1).to(device).long()}
            target_effectiveness = data[2].long().to(device)
            out = model(**composed_dict)
            loss = criterion_train(out, target_effectiveness)
            optimizer.zero_grad()
            loss.backward()

            fgm.attack()
            outt = model(**composed_dict)
            loss_sum = criterion_train(outt, target_effectiveness)
            loss_sum.backward()
            fgm.restore()

            optimizer.step()
            metric_acc_train.update(out.cpu(), target_effectiveness.cpu())
            metric_loss_train.update(loss.cpu())
            acc = metric_acc_train.compute()
            loss = metric_loss_train.compute()
            t.set_description("Epoch {:}/{:} Train".format(epoch, total_epochs))
            t.set_postfix(acc="{:.3f}".format(acc), loss="{:.3f}".format(loss))

            step += 1

            if step > T and (step - T) % T_interval == 1:
                ##
                if first == False:
                    model.eval()
                    with tqdm(eval_loader) as t1:
                        for data in t1:
                            with torch.no_grad():
                                composed_dict = {"input_ids": data[0].squeeze(1).to(device).long(),
                                                 "attention_mask": data[1].squeeze(1).to(device).long()}
                                target_effectiveness = data[2].long().to(device)
                                out = model(**composed_dict)
                                loss = criterion_eval(out, target_effectiveness)
                                metric_acc_eval.update(out.cpu(), target_effectiveness.cpu())
                                metric_loss_eval.update(loss.cpu())
                                acc = metric_acc_eval.compute()
                                loss = metric_loss_eval.compute()
                                t1.set_description("Epoch {:}/{:} Eval".format(epoch, total_epochs))
                                t1.set_postfix(acc="{:.3f}".format(acc), loss="{:.3f}".format(loss))
                    eval_acc = metric_acc_eval.compute()
                    eval_loss = metric_loss_eval.compute()
                    metric_acc_eval.reset()
                    metric_loss_eval.reset()
                    print(
                        "epoch: {}, evaluation loss: {:.3f}, evaluation acc: {:.3f}".format(epoch, eval_loss, eval_acc))
                    torch.save(model, "./deberta_model_pseudo_{:.3f}_CV{:}_{:}.pth".format(eval_loss, cv_id, step))

                    training_acc = metric_acc_train.compute()
                    training_loss = metric_loss_train.compute()
                    metric_acc_train.reset()
                    metric_loss_train.reset()

                    print("epoch: {}, iterations: {}, training loss: {:.3f}, training acc: {:.3f}".format(epoch, step,
                                                                                                          training_loss,
                                                                                                          training_acc))

                for idx, data in enumerate(unlabeled_loader):
                    if idx >= TT:
                        break
                    with torch.no_grad():
                        model.eval()
                        composed_dict = {"input_ids": data[0].squeeze(1).to(device),
                                         "attention_mask": data[1].squeeze(1).to(device)}
                        output_unlabeled = model(**composed_dict)
                        _, pseudo_labeled = torch.max(output_unlabeled, 1)
                    model.train()
                    out = model(**composed_dict)
                    unlabeled_loss = criterion_eval(out, pseudo_labeled)
                    t.set_postfix(acc="{:.3f}".format(acc), loss="{:.3f}".format(criterion_eval(out, pseudo_labeled)))
                    optimizer.zero_grad()
                    unlabeled_loss.backward()
                    optimizer.step()
                first = False
            scheduler.step()
