import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
from PIL import Image
import pandas as pd
from numpy import indices, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

"""

This script generates a dataset that contains x,y,z positions of each of 21 
points that are projected on a hand. Each hand gesture image will be translated into the 
array of 63 values. 

Such approach was used to reduce the dimensionality of the data. Instead of using
4000 pixel values (for 200px x 200px images) for training, we are now using only 63. 

"""

# For static images:
directory = 'datasets/images_labeled.csv'
data = pd.read_csv(directory)
data = data[78000:]

names = data['Names']
labels = data['Labels']

new_data = []
new_labels = []

NAMES = names

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence = 0.7,
    min_tracking_confidence = 0.7) as hands:
  
    for idx, name in enumerate(tqdm(NAMES)):
        label = labels[idx+78000]
        directory = '/Users/apple/Desktop/Studying/Python - Leon /FINNAL PROJECT/Image Classification on the American Sign Language Alphabet Dataset/asl_alphabet_train/{}/{}'.format(label,name)
        print(name)
        image = cv2.flip(cv2.imread(directory), 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
        if not results.multi_hand_landmarks:
            new_data.append([0]*63)
            new_labels.append(label)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                landmarks_per_iter = []

                new_labels.append(label)

                for lm in hand_landmarks.landmark:
                    landmarks_per_iter.append(lm.x)
                    landmarks_per_iter.append(lm.y)
                    landmarks_per_iter.append(lm.z)

                new_data.append(landmarks_per_iter)


landmarks_dataset = pd.read_csv('/Users/apple/Desktop/Studying/Python - Leon /FINNAL PROJECT/Image Classification on the American Sign Language Alphabet Dataset/landmarks_dataset_A3.csv')
old_labels = list(landmarks_dataset['Unnamed: 0'])
del landmarks_dataset["Unnamed: 0.1"]
del landmarks_dataset["Unnamed: 0"]
landmarks_dataset = np.array(landmarks_dataset)
new_data = np.array(new_data)
complete_data = np.append(landmarks_dataset, new_data, axis = 0)

labels_full_version = old_labels + new_labels

data_complete = pd.DataFrame(complete_data, index = labels_full_version)
print(data_complete)
directory = '/Users/apple/Desktop/Studying/Python - Leon /FINNAL PROJECT/Image Classification on the American Sign Language Alphabet Dataset/dataset_A5.csv'
data_complete.to_csv(directory)

failed_labels = []

for i in range(len(labels_full_version)):
    zeros = sum(x == 0 for x in data_complete.iloc[i][:])
    if zeros == 63:
        failed_labels.append(labels_full_version[i])

print("number of failed labels =", len(failed_labels))

sns.set_theme(style="darkgrid")
ax = sns.countplot(failed_labels)
plt.show()