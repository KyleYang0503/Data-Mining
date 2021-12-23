# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 16:13:48 2021

@author: Kyle
"""

import torch
from transformers import AdamW
from transformers import AutoModelForSequenceClassification
from dataset import CrossEncoderDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import random_split
import torch.nn.functional as F
from transformers import logging
logging.set_verbosity_warning()


def acc_(pred, label):
    pred = pred.detach()
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(dim=-1)
    return (pred == label).float().mean()


"""
PARAMETERS
"""

PRETRAINED_MODEL = 'roberta-large'
LEARN_RATE = 1e-5
BATCH_SIZE = 4
VAL_BATCH_SIZE = 8
ACCUMULATION_STEPS = 12
EPOCHS = 4
WARM_UP_RATE = 0.05
WARM_UP = True
VAILD = True
DEVICE = torch.device("cuda")


mode = 'train'

print("Currently Device:", DEVICE)


tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

model_name  = '110062621'
model_save_path = './outputs_models/110062621/'

log_fp = open(f'./logs/{model_name}.txt', 'w')

"""
Get Data
"""

train_set = CrossEncoderDataset(mode, tokenizer)
print('Data size: %d' %(len(train_set)))

train_set_size = int(len(train_set) * 0.9)
valid_set_size = len(train_set) - train_set_size
train_set, valid_set = random_split(train_set, [train_set_size, valid_set_size])
print(f'train_size : {train_set_size}, val_size {valid_set_size}')

valid_loader = DataLoader(valid_set, batch_size=VAL_BATCH_SIZE, shuffle=False)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

total_steps = len(train_loader) * EPOCHS / (BATCH_SIZE * ACCUMULATION_STEPS)
warm_up_steps = total_steps * WARM_UP_RATE


model = AutoModelForSequenceClassification.from_pretrained(PRETRAINED_MODEL, num_labels=8, ignore_mismatched_sizes=True)
optimizer = AdamW(model.parameters(), lr=LEARN_RATE)
scheduler = get_cosine_schedule_with_warmup(optimizer, warm_up_steps, total_steps)
loss_fct = nn.CrossEntropyLoss()

model = model.to(DEVICE)
model.train()

for EPOCHS in range(EPOCHS):
    running_loss = 0.0
    totals_batch = len(train_loader)
    acc = 0.0
    recall = 0.0
    f1 = 0.0
    precision = 0.0
    model.train()
    for i, data in enumerate(train_loader):        
        input_ids, attention_mask, labels = [t.to(DEVICE) for t in data]

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask,
                        labels=labels)
        logits = outputs.logits

        loss = loss_fct(logits, labels)
        running_loss += loss.item()
        loss = loss / ACCUMULATION_STEPS

        loss.backward()
        if ((i+1) % ACCUMULATION_STEPS) == 0 or ((i+1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        batch_acc = acc_(logits, labels)
        acc += batch_acc.detach().cuda()

        print(f'\r Epoch : {EPOCHS+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , end='' )
    print(f'Epoch : {EPOCHS+1}, batch : {i+1}/{totals_batch}, loss : {running_loss / (i+1) :.5f}, acc : {acc/ (i+1) :.5f}' , file=log_fp)
    print('')
    # valid 
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    totals_batch = len(valid_loader)
    for i, data in enumerate(valid_loader):        
        input_ids, attention_mask, labels = [t.to(DEVICE) for t in data]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            labels=labels)
            logits = outputs.logits

            loss = loss_fct(logits, labels)

        val_loss += loss.item()
        batch_acc = acc_(logits, labels)
        val_acc += batch_acc

        print(f'\r[val]Epoch : {EPOCHS+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}' , end='' )
    print(f'[val]Epoch : {EPOCHS+1}, batch : {i+1}/{totals_batch}, loss : {val_loss / (i+1) :.5f}, acc : {val_acc/ (i+1) :.5f}' , file=log_fp )
    log_fp.flush()
    torch.save(model.state_dict(), f"{model_save_path}/model_{str(EPOCHS+1)}.pt")
    print(' saved ')



























