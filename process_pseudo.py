import codecs

import pandas as pd
from transformers import AutoTokenizer

from utils import *

tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

INPUT_DIR = "./2021/"
text = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
text['essay_text'] = text['id'].apply(lambda x: get_essay(INPUT_DIR, x, is_train=True))

codecs.register_error("replace_encoding_with_utf8", replace_encoding_with_utf8)
codecs.register_error("replace_decoding_with_cp1252", replace_decoding_with_cp1252)

text['discourse_text'] = text['discourse_text'].apply(lambda x: resolve_encodings_and_normalize(x))
text['essay_text'] = text['essay_text'].apply(lambda x: resolve_encodings_and_normalize(x))
text["composed_text"] = text['discourse_type'] + ' ' + text['discourse_text'] + tokenizer.sep_token + text['essay_text']
text.to_csv("unlabeled_text.csv")
