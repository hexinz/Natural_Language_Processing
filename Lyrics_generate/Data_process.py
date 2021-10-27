#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import jsonlines
import pandas as pd
import numpy as np
import re


# # For LM training

# ## Training data

# In[70]:


with jsonlines.open("song-lyrics.train.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if len(obj['lyrics']) > 500:
            continue
        if num == 100000:
            break
        with open("final_data/train.txt","a") as write_f:
            num += 1
            temp = obj['lyrics'].replace('\n', ' ñ ')
            write_f.write("{}\n".format(temp))


# ## dev data

# In[71]:


with jsonlines.open("song-lyrics.test.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if len(obj['lyrics']) > 500:
            continue
        if num == 10000:
            break
        with open("final_data/test.txt","a") as write_f:
            num += 1
            temp = obj['lyrics'].replace('\n', ' ñ ')
            write_f.write("{}\n".format(temp))


# ## testing data

# In[72]:


with jsonlines.open("song-lyrics.dev.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if len(obj['lyrics']) > 500:
            continue
        if num == 10000:
            break
        with open("final_data/dev.txt","a") as write_f:
            num += 1
            temp = obj['lyrics'].replace('\n', ' ñ ')
            temp = temp.replace('\n', ' newline ')
            write_f.write("{}\n".format(temp))


# # For classification

# ## Training data

# In[73]:


text = []
label = []


with jsonlines.open("song-lyrics.train.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 10000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        label.append(0)


with jsonlines.open("lyrics.machine-gen.train.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 10000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        
        label.append(1)

result = {'labels': label, 'text':text}

pd.DataFrame(result).to_csv('classi_data/train.csv', index=False)


# ## dev data

# In[74]:


text = []
label = []


with jsonlines.open("song-lyrics.dev.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 1000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        label.append(0)


with jsonlines.open("lyrics.machine-gen.dev.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 1000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        label.append(1)

result = {'labels': label, 'text':text}

pd.DataFrame(result).to_csv('classi_data/dev.csv', index=False)


# ## testing data

# In[75]:


text = []
label = []


with jsonlines.open("song-lyrics.test.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 1000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        label.append(0)


with jsonlines.open("lyrics.machine-gen.test.jsonl",'r') as load_f:
    num = 0
    for obj in load_f:
        if num == 1000:
            break
        num += 1
        text.append(obj['lyrics'].replace('\n', ' ñ '))
        label.append(1)

result = {'labels': label, 'text':text}

pd.DataFrame(result).to_csv('classi_data/test.csv', index=False)


# ## self-generated lyrics

# In[72]:


text = []
label = []

with open("500lyrics.json",'r') as load_f:
    result = json.load(load_f)
    for item in result:
        text.append(item['lyrics'])
        label.append(1)
        
result = {'labels': label, 'text':text}

pd.DataFrame(result).to_csv('classi_data/500lyrics.csv', index=False)







