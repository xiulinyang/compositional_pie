import pandas as pd
from typing import List
import torch
import transformers
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
import numpy as np
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=False)
model = GPT2LMHeadModel.from_pretrained('gpt2')
bos_id = model.config.bos_token_id
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

def preprocess(tokens):
  inputs = tokenizer(tokens, is_split_into_words=True)
  model_inputs = transformers.BatchEncoding({
    "input_ids": torch.tensor([bos_id] + inputs.input_ids).unsqueeze(0),
    "attention_mask": torch.tensor([1] + inputs.attention_mask).unsqueeze(0)
})
  with torch.no_grad(): # disable gradient computation
    outputs = model(**model_inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    probs = probs.squeeze(0)
    surprisals = -1 * torch.log2(probs)
    surprisals = surprisals[:-1]
    token_ids = model_inputs.input_ids.squeeze(0)[1:]
    tokens = tokenizer.convert_ids_to_tokens(model_inputs.input_ids.squeeze(0))[1:]
    token_surprisal = surprisals[range(len(surprisals)), token_ids]
    index = torch.arange(0, token_ids.shape[0])
    surp = -1 * torch.log2(F.softmax(outputs.logits, dim=-1).squeeze(0)[index, token_ids])
    tokens = tokenizer.convert_ids_to_tokens(token_ids.squeeze())
    return tokens, token_surprisal

def get_word_surprisal(tokens: List[str], token_surprisal: List[float]) -> pd.DataFrame:

    word_surprisal = []
    words = []

    i = 0
    temp_token = ""
    temp_surprisal = 0

    while i <= len(tokens)-1:

        temp_token += tokens[i]
        temp_surprisal += token_surprisal[i]

        if i == len(tokens)-1 or tokens[i+1].startswith("Ġ"):
            # remove start-of-token indicator
            words.append(temp_token[1:])
            word_surprisal.append(temp_surprisal)
            # reset temp token/surprisal
            temp_surprisal = 0
            temp_token = ""
        i +=1
    # print(words)
    word_surprisal = [t.item() for t in word_surprisal]
    # print(word_surprisal)
    return words[-1], word_surprisal[-1]

e=0
i_surprisals = []
l_surprisals = []
with open('class_pies_C_new.json') as pies:
  pie_files = json.load(pies)
  surprisals = {key:[] for key in pie_files}
  for key, value in pie_files.items():
    for v in value:
      try:
        if v:
          verb = v[0]
          # print(verb)
          noun = v[2]
          # print(noun)
          # print(noun)
          verb_tokens, verb_surprisals = preprocess(verb)
          noun_tokens, noun_surprisals = preprocess(noun)

          v, verb_s = get_word_surprisal(verb_tokens, verb_surprisals)
          n, noun_s = get_word_surprisal(noun_tokens, noun_surprisals)
          # print(v, n)

          difference = verb_s - noun_s
          # print(type(difference))
          # print(difference)
          if key[-1]=='I':
            i_surprisals.append(difference)
          else:
            l_surprisals.append(difference)
      except:
        e+=1
        print('error')
        print(e)

# sorted_keys = sorted(surprisals)
# surprisals = {key: surprisals[key] for key in sorted_keys}

print(len(i_surprisals))
print(len(l_surprisals))
mean_surprisal_i = np.mean(i_surprisals)
mean_surprisal_l = np.mean(l_surprisals)
median_surprisal_i = np.median(i_surprisals)
median_surprisal_l = np.median(l_surprisals)
variance_surprisal_i = np.std(i_surprisals)
variance_surprisal_l = np.std(l_surprisals)

# Print statistics
print(f'mean_surprisal_i: {mean_surprisal_i}')
print(f'mean_surprisal_l: {mean_surprisal_l}')
print(f'median_surprisal_i: {median_surprisal_i}')
print(f'median_surprisal_l: {median_surprisal_l}')
print(f'variance_surprisal_i: {variance_surprisal_i}')
print(f'variance_surprisal_l: {variance_surprisal_l}')

# Create box plots using seaborn or matplotlib
plt.figure(figsize=(10, 6))
sns.boxplot(data=[i_surprisals, l_surprisals])
plt.xticks([0, 1], ['Idiom Surprisals', 'Literal Surprisals'])
plt.title('Box plot of Surprisal Values')
plt.show()
plt.savefig('surprisal_boxplot.png')

plt.close()

# Normality test
_, p_value_i = stats.shapiro(i_surprisals)
_, p_value_l = stats.shapiro(l_surprisals)
print("Normality Test p-values: Idioms =", p_value_i, ", Literals =", p_value_l)

# Equal variances test
_, p_value_var = stats.levene(i_surprisals, l_surprisals)
print("Equal Variance Test p-value:", p_value_var)

if p_value_var > 0.05:  # Equal variances
    U1, p_value = mannwhitneyu(i_surprisals, l_surprisals)

else:  # Unequal variances (Welch’s t-test)
    t_stat, p_value = scipy.stats.ttest_ind(i_surprisals, l_surprisals, equal_var=False)

n1 = len(i_surprisals)
n2 = len(l_surprisals)
U2 = n1*n2-U1
print(f'U2{U2}')
u_stat = min(U1, U2)
# Calculate the rank-biserial correlation as effect size
r = 1 - (2 * u_stat) / (n1 * n2)
print("Effect size (rank-biserial correlation):", r)
print("U:", u_stat, "p-value:", p_value)
# Interpretation
if p_value < 0.05:
    print("Significant difference between groups")
else:
    print("No significant difference between groups")
