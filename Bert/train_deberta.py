import warnings

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModel, AutoConfig

from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from torch.optim import AdamW
import torchmetrics
from utils import *
from transformers import get_cosine_schedule_with_warmup
import pickle

device = torch.device('cuda:{:}'.format(0))


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
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


cv_id = 0

# 18 for 24G
batch_size = 2
warmup_ratio = 0.1

rate_decay = 0.9
base_rate = 1e-5

weight_decay = 0.01
total_epochs = 5

threshold = 0.8

criterion_train = FocalLoss(gamma=0.5)
criterion_eval = nn.CrossEntropyLoss()

train_text = pickle.load(open("../cv{:}.train.pkl".format(cv_id), "rb"))
eval_text = pickle.load(open("../cv{:}.eval.pkl".format(cv_id), "rb"))
unlabeled_text = pd.read_csv("../unlabeled_text.csv")

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')
tokenizer.model_max_length = 512

model = FeedBackModel('microsoft/deberta-v3-base')
model.model = AutoModel.from_pretrained("microsoft/deberta-v3-base")
model.to(device)
fgm = FGM(model)

train_loader = DataLoader(Feedback_Data(train_text),
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=0, drop_last=False)

eval_loader = DataLoader(Feedback_Data(eval_text),
                         batch_size=batch_size,
                         shuffle=False,
                         pin_memory=False,
                         num_workers=0, drop_last=False)

unlabeled_loader = DataLoader(Feedback_Data_unlabeled(unlabeled_text, tokenizer),
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=False,
                              num_workers=0, drop_last=False)

num_steps = len(train_loader) * total_epochs

optimizer_grouped_parameters = [{'params': p, 'lr': base_rate * (rate_decay ** (12 - int(n.split(".")[3])))} for n, p in
                                model.named_parameters() if n.split(".")[1] == "encoder" and len(n.split(".")) > 4] + \
                               [{'params': [p for n, p in model.named_parameters() if n.split(".")[1] == "embeddings"],
                                 'lr': base_rate * (rate_decay ** 12)}]

optimizer = AdamW(optimizer_grouped_parameters, lr=base_rate, weight_decay=weight_decay)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(num_steps * warmup_ratio),
                                            num_training_steps=num_steps)

metric_acc = torchmetrics.Accuracy().cpu()
metric_loss = torchmetrics.MeanMetric().cpu()

for epoch in range(1, total_epochs + 1):
    model.train()
    with tqdm(train_loader) as t:
        for data in t:
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
            metric_acc.update(out.cpu(), target_effectiveness.cpu())
            metric_loss.update(loss.cpu())
            t.set_description("Epoch {:}/{:} Train".format(epoch, total_epochs))
            t.set_postfix(acc="{:.3f}".format(metric_acc.compute()), loss="{:.3f}".format(metric_loss.compute()))

            scheduler.step()
    metric_acc.reset()
    metric_loss.reset()

    with tqdm(unlabeled_loader) as t:
        for data in unlabeled_loader:
            model.eval()
            with torch.no_grad():
                composed_dict = {"input_ids": data[0].squeeze(1).to(device),
                                 "attention_mask": data[1].squeeze(1).to(device)}
                output_unlabeled = model(**composed_dict)
                pseudo_proba, pseudo_labeled = torch.max(output_unlabeled, 1)
            mask = pseudo_proba > threshold
            model.train()
            out = model(**composed_dict)
            unlabeled_loss = criterion_eval(out[mask], pseudo_labeled[mask])
            metric_loss.update(unlabeled_loss.cpu())
            t.set_postfix(loss="{:.3f}".format(metric_loss.compute()))
            optimizer.zero_grad()
            unlabeled_loss.backward()
            optimizer.step()
    metric_loss.reset()

    model.eval()
    with tqdm(eval_loader) as t:
        for data in t:
            with torch.no_grad():
                composed_dict = {"input_ids": data[0].squeeze(1).to(device).long(),
                                 "attention_mask": data[1].squeeze(1).to(device).long()}
                target_effectiveness = data[2].long().to(device)
                out = model(**composed_dict)
                loss = criterion_eval(out, target_effectiveness)
                metric_acc.update(out.cpu(), target_effectiveness.cpu())
                metric_loss.update(loss.cpu())
                t.set_description("Epoch {:}/{:} Eval".format(epoch, total_epochs))
                t.set_postfix(acc="{:.3f}".format(metric_acc.compute()), loss="{:.3f}".format(metric_loss.compute()))
    metric_acc.reset()
    metric_loss.reset()

    torch.save(model, "./deberta_model_pseudo_{:.3f}_CV{:}_{:}.pth".format(loss, cv_id, epoch))
