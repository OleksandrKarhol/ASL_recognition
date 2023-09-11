import numpy
import cv2
from PIL import Image
import pandas as pd
from numpy import random 
import string
from  tqdm import tqdm

max_n = list(range(3001))[1:]

alphabet = list(string.ascii_uppercase) 
alphabet.append('del')
alphabet.append('space')

image_names = []
labels = []

for a in tqdm(alphabet):
    for n in max_n:
        directory = 'asl_alphabet_train/{}/{}{}.jpg'.format(a, a, n)
        image = Image.open(directory)
        image_name = '{}{}.jpg'.format(a, n)
        image_names.append(image_name)
        labels.append(a)

image_data = pd.DataFrame(list(zip(image_names,labels)), columns = ['Names', 'Labels'])
image_data.to_csv('datasets/images_labeled.csv')