#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install simpletransformers


# # Part 1

# In[2]:


import pandas as pd
import random
from collections import defaultdict, Counter
from tqdm.notebook import tqdm
# from simpletransformers.classification import ClassificationModel, ClassificationArgs
import logging
from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)
import numpy as np
from sklearn.metrics import f1_score
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[4]:


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = LanguageModelingArgs()
model_args.reprocess_input_data = True
model_args.num_train_epochs = 5
model_args.dataset_type = "simple"
model_args.mlm = False  # mlm must be False for CLM
model_args.best_model_dir = '.'
model_args.evaluate_during_training = True
model_args.overwrite_output_dir = True
model_args.train_batch_size = 2
model_args.eval_batch_size = 2

# model = LanguageModelingModel(
#     "gpt2", "gpt2-medium", args=model_args, use_cuda=False
# )

model = LanguageModelingModel(
    "gpt2", "gpt2-medium", args=model_args, use_cuda=True
)


# In[11]:


train_file = "./drive/MyDrive/train.txt"
test_file = "./drive/MyDrive/test.txt"
dev_file = "./drive/MyDrive/dev.txt"


# In[ ]:


model.train_model(train_file, eval_file=dev_file)


# In[ ]:


result_train = model.eval_model(train_file)
result_dev = model.eval_model(dev_file)
result_test = model.eval_model(test_file)


# # Part 2

# In[3]:


import logging
from simpletransformers.language_generation import LanguageGenerationModel

from simpletransformers.language_generation import (
    LanguageGenerationModel,
    LanguageGenerationArgs,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = LanguageGenerationArgs()

model_args.max_length = 20

model = LanguageGenerationModel(
    "gpt2",
    "checkpoint-10000",
    args=model_args,
    use_cuda=False
)

prompts = [
    "My",
    "The",
    "One",
    "When",
    "If"
]

for prompt in prompts:
    # Generate text using the model. Verbose set to False to prevent logging generated sequences.
    generated = model.generate(prompt)
    result = '\n'.join(generated[0].split(' ñ ')) + '\n'
    result = result.replace(' ñ','')
    result = result.replace('\n\n','\n')
    print(result)


# In[7]:


import json
result_json = []
with open("prompts_origin.txt","r") as read_f:
    num = 0
    for f in read_f:
        generated = model.generate(f[:-1])
        result = '\n'.join(generated[0].split(' ñ ')) + '\n'
        result = result.replace(' ñ','')
        result = result.replace('\n\n','\n')
        lyrics = {"id": num, "lyrics": result}
        result_json.append(lyrics)
with open ('500lyrics.json', 'w') as fout :
    json.dump(result_json, fout, indent =4)


# # Part 3

# In[ ]:


train_file = "train.csv"
test_file = "test.csv"
dev_file = "dev.csv"
my_file = "500lyrics.csv"

train_df = pd.read_csv(train_file)
test_df = pd.read_csv(test_file)
dev_df = pd.read_csv(dev_file)
my_df = pd.read_csv(my_file)


# In[ ]:


from simpletransformers.classification import ClassificationModel, ClassificationArgs
model_args = ClassificationArgs(num_train_epochs=1)


# Note here that we're saying to use CUDA, which has the model run on a GPU. If you don't have
# a GPU, set this to False.
model = ClassificationModel('distilbert', 'distilbert-base-uncased', args=model_args, use_cuda=False)


# Train the model for a single epoch (a single pass through the data).
# On my laptop with no GPU this would take around 90 minutes.
model.train_model(train_df)



# In[ ]:


# Evaluate the model. We can pass in specific metrics that we want to use (e.g., the f1_score) as kwargs 
result, model_outputs, wrong_predictions = model.eval_model(train_df, f1=f1_score)


# The results contain all the built-in metrics, as well as our kwarg metric
print(result)


# In[ ]:


# Evaluate the model. We can pass in specific metrics that we want to use (e.g., the f1_score) as kwargs 
result, model_outputs, wrong_predictions = model.eval_model(dev_df, f1=f1_score)


# The results contain all the built-in metrics, as well as our kwarg metric
print(result)


# In[ ]:


# Evaluate the model. We can pass in specific metrics that we want to use (e.g., the f1_score) as kwargs 
result, model_outputs, wrong_predictions = model.eval_model(dev_df, f1=f1_score)


# The results contain all the built-in metrics, as well as our kwarg metric
print(result)


# In[ ]:


# Evaluate the model. We can pass in specific metrics that we want to use (e.g., the f1_score) as kwargs 
result, model_outputs, wrong_predictions = model.eval_model(my_df, f1=f1_score)


# The results contain all the built-in metrics, as well as our kwarg metric
print(result)

