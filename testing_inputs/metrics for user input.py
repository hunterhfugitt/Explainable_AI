from mtgsdk import Card
from mtgsdk import Set
from mtgsdk import Type
from mtgsdk import Supertype
from mtgsdk import Subtype
from mtgsdk import Changelog
import os
import nltk
import regex as re
from nltk.tokenize import word_tokenize
import pickle
from PIL import Image, ImageDraw, ImageFont
import math
from scipy import spatial
import numpy as np
import requests
from tkinter import Y
import PIL
import io
import urllib
import json
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from scipy import spatial
import sklearn 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.cluster import KMeans
import networkx as nx
from sklearn.decomposition import PCA
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transformsgit 
import pandas as pd
import numpy as np
from datetime import datetime
from random import sample
import umap.umap_ as umap
import hdbscan
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import sys
from bs4 import BeautifulSoup
import random
import glob

list_values = [0,0.2,0.4,0.6,0.8,1.0]
data_frame = pd.DataFrame()

place =  os.getcwd()                   
#create relevant lists

dict_happened = {}
original_string = []
color_input = []
type_input = []
text_input = []
flavor_input = []
color_score = []
type_score = []
text_score = []
flavor_score = []

#find the sum of values
for a in list_values:
    for b in list_values:
        for c in list_values:
            for d in list_values:
                if a != 0 or b!= 0 or c !=0 or d!= 0:
                    string = f'co{a},ty{b},te{c},fl{d}'
                    file = open(f'{place}\\static\\test_user_input\\{string}_list.pickle','rb')
                    list_to_add = pickle.load(file)
                    file.close()
                    if f'{a}{b}{c}{d}' in dict_happened:
                        pass
                    else:
                        dict_happened[f'{a}{b}{c}{d}'] = True
                        original_string.append(string)
                        color_input.append(a)
                        type_input.append(b)
                        text_input.append(c)
                        flavor_input.append(d)
                        list_to_add = pickle.load(file)
                        list_before = list_to_add[0]
                        list_after = list_to_add[2]
                        sum = 0.0
                        for count,each in enumerate(list_before):
                            difference  = spatial.distance.cosine(each[0],list_after[count][0])
                            sum = sum + difference
                        color_score.append(sum)
                        sum = 0.0
                        for count,each in enumerate(list_before):
                            difference  = spatial.distance.cosine(each[1],list_after[count][1])
                            sum = sum + difference
                        type_score.append(sum)
                        sum = 0.0
                        for count,each in enumerate(list_before):
                            difference  = spatial.distance.cosine(each[2],list_after[count][2])
                            sum = sum + difference
                        text_score.append(sum)
                        sum = 0.0
                        for count,each in enumerate(list_before):
                            difference  = spatial.distance.cosine(each[3],list_after[count][3])
                            sum = sum + difference
                        flavor_score.append(sum)

#input into dataframe
                                        

