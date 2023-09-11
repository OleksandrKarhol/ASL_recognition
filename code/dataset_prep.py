import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
from numpy import random
import string

"""

This script deletes the images that were not successfully translated into the arrays of 
63 numeric values aka x,y,z positions of each of 21 points that are projected on a hand 

"""

direcory = ('datasets/dataset.csv')
dataset = pd.read_csv(direcory)

failed_labels = []

for i in range(len(dataset)):
    zeros = sum(x == 0 for x in dataset.iloc[i][:])
    if zeros == 63:
       failed_labels.append(i)

print("number of failed labels =", len(failed_labels))

new_dataset = dataset.drop(labels=failed_labels, axis=0)

direcory = ('datasets/dataset_cleaned.csv')
new_dataset.to_csv(direcory)
print(len(new_dataset), len(dataset))

labels = new_dataset['Labels']
sns.set_theme(style="darkgrid")
ax = sns.countplot(labels)
plt.show()