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

with open('class_pies_C_new.json', 'r') as c:
    categories = json.load(c)

i_surprisals = []
l_surprisals = []
for key, value in categories.items():
    for v in value:
        if v:
            surprisal_value_v = surprisal(v[0], v[1])
            surprisal_value_n = surprisal(v[2], v[3])
            difference = surprisal_value_v-surprisal_value_n
            if key[-1]=='I':
                i_surprisals.append(difference)
            else:
                l_surprisals.append(difference)

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
print(f'sd_surprisal_i: {variance_surprisal_i}')
print(f'sd_surprisal_l: {variance_surprisal_l}')

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

# Perform t-test
if p_value_var > 0.05:  # Equal variances
    U1, p_value = mannwhitneyu(i_surprisals, l_surprisals)
    
else:  # Unequal variances (Welch’s t-test)
    print('aaaa')
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
