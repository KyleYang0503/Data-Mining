# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 12:25:52 2021

@author: Kyle
"""

import torch
from transformers import AutoModelForSequenceClassification
from dataset import CrossEncoderDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
from transformers import logging
import pandas as pd
logging.set_verbosity_warning()
### hyperparams ###

# pretrained_model = 'cardiffnlp/twitter-roberta-base-emotion'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


"""
PARAMETERS
"""

pretrained_model = 'roberta-large'
batch_size = 256
mode = 'test'
idx2emotion = {0: 'anticipation', 1: 'sadness', 2: 'fear', 3: 'joy', 4: 'anger', 5: 'trust', 6: 'disgust', 7: 'surprise'}


tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

model_name = pretrained_model.split('/')[-1]
model_name = '110062621'
model_save_path = './trained_models/%s/model_4.pt'%model_name
print(model_save_path)
test_set = CrossEncoderDataset(mode, tokenizer)
print("testing_data size:%d"%(len(test_set)))


"""
Get Data
"""
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


model = AutoModelForSequenceClassification.from_pretrained(pretrained_model, num_labels=8, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(model_save_path), strict=False)
model = model.to(device)
model.eval()

all_ids = []
all_preds = []

with torch.no_grad():
    totals_batch = len(test_loader)
    for i, data in enumerate(test_loader):        
        input_ids, attention_mask = [t.to(device) for t in data[0]]

        tweet_ids = data[1]
        all_ids += list(tweet_ids)
        
        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask)
        logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1)

        all_preds += list(preds)
        print(f'\r batch : {i+1}/{totals_batch}' , end='' )


all_emotions = []
for pred in all_preds:
    all_emotions.append(idx2emotion[pred])
    
df = pd.DataFrame(
    {'id': all_ids,
     'emotion': all_emotions,
    })

df.to_csv(f'./{model_name}.csv', index=False)
