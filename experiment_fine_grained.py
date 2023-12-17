import math
import kenlm
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
model = kenlm.LanguageModel('bnc.binary')

def surprisal(sentence_with, sentence_without):
    log_with = model.score(sentence_with)
    log_without = model.score(sentence_without)
    surprisal = log_without-log_with
    surprisal = surprisal/math.log(2)
    return surprisal

with open('class_pies_C_new.json') as c:

    categories = json.load(c)

surprisals = {key:[] for key in categories}
for key, value in categories.items():
    for v in value:
        if v:
            surprisal_value_v = surprisal(v[0], v[1])
            surprisal_value_n = surprisal(v[2], v[3])
            #difference = surprisal_value_v-surprisal_value_n
            
            surprisals[key].append(len(v[0].split()))
sorted_keys = sorted(surprisals)
surprisals = {key: surprisals[key] for key in sorted_keys}

for key, values in surprisals.items():
    print(key)
    
    mean_surprisal = np.mean(values)
    median_surprisal = np.median(values)
    variance_surprisal = np.std(values)
    print(f'{key}:{mean_surprisal}')
    print(f'{key}:{median_surprisal}')
    print(f'{key}:{variance_surprisal}')


data = [surprisals[key] for key in sorted_keys]

palette = sns.color_palette("husl", len(sorted_keys) // 2)
colors = [color for color in palette for _ in (0, 1)]  # Duplicate each color once

# Creating the box plot
plt.figure(figsize=(15, 10))
box = plt.boxplot(data, patch_artist=True)

# Applying colors to each pair of box plots
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Adding labels, title, etc.
plt.xticks(range(1, len(sorted_keys) + 1), sorted_keys, rotation=45)
plt.xlabel('Group')
plt.ylabel('Surprisal Value')
plt.title('Box Plot of Surprisal Diffrence by Group')
plt.grid(True)

plt.show()
plt.savefig('surprisal_boxplot_by_group.png')

plt.close()
