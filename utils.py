import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from text_unidecode import unidecode
from torch.autograd import Variable
from torch.utils.data import Dataset


def get_essay(INPUT_DIR, essay_id, is_train=True):
    parent_path = INPUT_DIR + 'train' if is_train else INPUT_DIR + 'test'
    essay_path = os.path.join(parent_path, f"{essay_id}.txt")
    essay_text = open(essay_path, 'r').read()
    return essay_text


def replace_encoding_with_utf8(error: UnicodeError) -> Tuple[bytes, int]:
    return error.object[error.start: error.end].encode("utf-8"), error.end


def replace_decoding_with_cp1252(error: UnicodeError) -> Tuple[str, int]:
    return error.object[error.start: error.end].decode("cp1252"), error.end


def resolve_encodings_and_normalize(text: str) -> str:
    """Resolve the encoding problems and normalize the abnormal characters."""
    text = (
        text.encode("raw_unicode_escape")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
        .encode("cp1252", errors="replace_encoding_with_utf8")
        .decode("utf-8", errors="replace_decoding_with_cp1252")
    )
    text = unidecode(text)
    return text


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, -1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Feedback_Data(Dataset):
    def __init__(self, df):
        self.df = df

    def __getitem__(self, idx):
        return self.df[idx]

    def __len__(self):
        return len(self.df)


class Feedback_Data_unlabeled(Dataset):
    def __init__(self, df, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        composed_text, _, composed_text_mask = self.tokenizer.encode_plus(self.df.iloc[idx]["composed_text"],
                                                                          padding='max_length', truncation=True,
                                                                          add_special_tokens=True,
                                                                          return_tensors='pt').values()
        return composed_text, composed_text_mask

    def __len__(self):
        return len(self.df)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name.split(".")[1]:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name == name.split(".")[1]:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
