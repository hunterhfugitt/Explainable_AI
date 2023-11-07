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

stop_words = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[\w\']+')
directory = os.getcwd()


#Auto encoder info from here: https://towardsdatascience.com/deep-clustering-with-sparse-data-b2eb1bf2922e
def get_conv_autoencoder(input_shape, filters, kernel_size, act='relu'):
    # Number of encoding and decoding layers
    n_stacks = len(filters) - 1
    weights_input = Input(shape=(4,), name='weights_input')

    # Input layer
    x = Input(shape=input_shape, name='input')
    h = x
    
    # Encoding layers
    for i in range(n_stacks):
        h = Conv2D(filters[i], kernel_size, activation=act, padding='same', name='encoder_%d' % i)(h)

    # Flatten the tensor output of the last encoding layer
    h = Flatten()(h)
    h = Dense(filters[-1], name='encoder_%d' % n_stacks)(h)

    # Reshape the flat tensor to prepare for the decoding layers
    h = Dense(np.prod(input_shape), name='decoder_dense')(h)
    h = Reshape(input_shape, name='decoder_reshape')(h)

    # Decoding layers
    for i in range(n_stacks, 0, -1):
        h = Conv2DTranspose(filters[i-1], kernel_size, activation=act, padding='same', name='decoder_%d' % i)(h)

    # Output layer (reconstructed image)
    h = Conv2DTranspose(input_shape[-1], kernel_size, padding='same', name='decoder_0')(h)
    
    # Create and return the model
    model = Model(inputs=x, outputs=h)
    model.summary()
    
    return model

# def loss_list(y_true, y_pred):
#     length = 0 
#     list_loss = []
#     loss_total = 0
#     list_values = [0,1,2,3]
#     for count,each in enumerate(list_values):
#         loss = tf.reduce_mean(tf.abs(y_true[count] - y_pred[count]))
#         list_loss.append(loss)
#     return(list_loss)

# class LossList(Metric):
#     def __init__(self, name="loss_list", **kwargs):
#         super(LossList, self).__init__(name=name, **kwargs)
#         self.loss_list = self.add_weight(name="ll", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         loss = tf.reduce_mean(tf.square(y_true - y_pred))
#         self.loss_list.assign(loss)

#     def result(self):
#         return self.loss_list

#     def reset_states(self):
#         # The state of the metric will be reset at the start of each epoch.
#         self.loss_list.assign(0.)

lowest_loss = 999999999
loss_list = []

# def loss_list(y_true, y_pred):
#     loss_list_current = []
#     value1 = K.mean(K.abs(y_true[0] - y_pred[0]))
#     value2 = K.mean(K.abs(y_true[1] - y_pred[1]))
#     value3 = K.mean(K.abs(y_true[2] - y_pred[2]))
#     value4 = K.mean(K.abs(y_true[2] - y_pred[3]))
#     loss_list_current.append(value1)
#     loss_list_current.append(value2)
#     loss_list_current.append(value3)
#     loss_list_current.append(value4)
#     return loss_list_current

def create_weighted_loss(weights_input):
    def weighted_loss(y_true, y_pred):
        length = 0 
        weights = weights_input
        print("Shape of y_true: ", y_true.shape)
        print("Shape of y_pred: ", y_pred.shape)
        #loss_list_local = []
        loss_total = 0
        for count,each in enumerate(weights_input):
            length = length + 1
            loss = tf.reduce_mean(tf.abs(y_true[count] - y_pred[count]))
            #loss_list_local.append(loss)
            scaled_loss = loss * each
            loss_total = scaled_loss + loss_total
        # global lowest_loss
        # global loss_list
        # if(loss_total <= lowest_loss):
        #     lowest_loss = loss_total
        #     loss_list = loss_list_local
        # Return the mean of the weighted errors
        return (loss_total)
    
    return weighted_loss


def learn_manifold(x_data, umap_min_dist=0.00, umap_metric='euclidean', umap_dim=10, umap_neighbors=30):
    md = float(umap_min_dist)
    print(x_data)
    try:
        return umap.UMAP(
            random_state=0,
            metric=umap_metric,
            n_components=umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=md).fit_transform(x_data)
    except:
        umap_neighbors=10
        umap_dim=10
        return umap.UMAP(
            random_state=0,
            metric=umap_metric,
            n_components=umap_dim,
            n_neighbors=umap_neighbors,
            min_dist=md).fit_transform(x_data)
        
def NN_Put_Together(X,weights,model_hash,batch,epochs,filters,kernal):
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=11)
    batch_size = batch
    pretrain_epochs = epochs
    #shape = [X.shape[-1], 2500, 500, 1000, encoded_dimensions]
    shapeD1 = X.shape[-2]
    shapeD2 = X.shape[-1]
    #shape = [X.shape[-1], 1000, 1000, 1500, 250, 250, 500, encoded_dimensions]
    #shape = [X.shape[-1], 1000, 1000, 1500, 750, 750, 500, 250, 250, 500, encoded_dimensions]
    shape = (shapeD1,shapeD2,1)
    autoencoder = get_conv_autoencoder(shape,filters,kernal,act='relu',)
    encoded_layer = f'encoder_{len(filters) - 1}'
    second_layer = 'encoder_0'
    decoder_layer = 'decoder_0'
    hidden_decoder_layer = autoencoder.get_layer(name=decoder_layer).output
    hidden_second_layer = autoencoder.get_layer(name=second_layer).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden_decoder_layer)
    autoencoder.compile(loss=create_weighted_loss(weights), optimizer='adam')
    #train the autoencoder
    model_series = f'CLS_MODEL_{model_hash}' + datetime.now().strftime("%h%d%Y-%H%M")

    checkpointer = ModelCheckpoint(filepath=f"{model_series}-model.h5", verbose=0, save_best_only=True)
    X_CNN = X.reshape(X.shape[-3],shapeD1,shapeD2,1)
    autoencoder.fit(
        X_train,
        X_train,
        batch_size=batch_size,
        epochs=pretrain_epochs,
        verbose=1,
        validation_data=(X_test, X_test),
        callbacks=[checkpointer]
    )
    auto_encoded_text_vector = {}
    with tf.keras.utils.custom_object_scope({'weighted_loss': create_weighted_loss(weights)}):
        autoencoder = load_model(f"{model_series}-model.h5")
    #autoencoder.save(f"{model_hash}_my_model")
    #autoencoder.save_weights(f"{model_hash}_my_model_weights")
    input_shape = autoencoder.input_shape
    X_encoded = encoder.predict(X_CNN)
    num_samples = X_CNN.shape[0]
    print('this_encoded')
    auto_encoded_text_vector = {}
    directory = os.getcwd()
    full_path = os.path.join(directory, f'*{model_series}*')
    matching_files = glob.glob(full_path)
    #loss, loss_list = autoencoder.evaluate(X_test, X_encoded)
    for file_name in matching_files:
        os.remove(file_name)
    return(X_encoded)


def nn_me(vector,X,weights,saved_name):
    print(f'This is the shape{X.shape}')
    annotations = vector
    count2 = 0
    #Change values of neural network encoding here
    batches = 128
    epochs = 64
    encoded_dimensions = 50
    filters = [8,16,8,encoded_dimensions]
    kernal = (3,3)
    X_decoded = NN_Put_Together(X,weights,saved_name,batches,epochs,filters,kernal)
    # X_encoded = NN_Put_Together(X,weights,saved_name,batches,epochs,filters,kernal)
    #X_reduced = learn_manifold(X_encoded, umap_neighbors=30, umap_dim=int(encoded_dimensions/2))
    # reducer = umap.UMAP(n_components=2)
    # embedding = reducer.fit_transform(X_encoded)
    # list_X = []
    # list_Y = []
    # for each in embedding:
    #     list_X.append(each[0])
    #     list_Y.append(each[1])
    # smaller = X_encoded
    # count = 0
    # check = True
    # dir = os.getcwd()+'\\static\\images\\card_images'
    # current = os.getcwd()
    # return(smaller)
    list_values = []
    for count,each in enumerate(X_decoded):
        list_values.append([])
        for count2,smaller_list in enumerate(each):
            changed_list = []
            for even_smaller_list in smaller_list:
                #print(even_smaller_list)
                changed_list.append(even_smaller_list[0])
            list_values[count].append(changed_list)
            #X_decoded[count][count2] = changed_list
    list_values = np.array(list_values)
    return(list_values)



def _Compare_all_cards(vector,card1,which_list):
    list_card_comparison = []
    for count,key in enumerate(vector):
        if key != card1:
            list_card_comparison.append([])
            list_card_comparison[count].append(key)
            list_card_comparison[count].append(_Compare_cards(vector,card1,key,which_list))
        else:
            list_card_comparison.append([])
            list_card_comparison[count].append(key)
            list_card_comparison[count].append(99999)
    list_card_comparison =  sorted(list_card_comparison, reverse = False, key=lambda kv: kv[:][1])
    return(list_card_comparison)

            
def _cluster_this(new_scaling,scaling1,scaling2,scaling3,scaling4,list1,list2,list3,cached_value):
    global lowest_loss
    # scaling1 = new_scaling[0] *scaling1
    # scaling2 = new_scaling[1] * scaling2
    # scaling3 = new_scaling[2] * scaling3
    # scaling4 = new_scaling[3] * scaling4
    lowest_loss = 999999999
    global loss_list
    loss_list = []
    list_keys = []
    dict_large = {}
    list_vector = []
    place = os.getcwd()
    print(place)
    used = []
    color_feature = scaling1
    type_feature = scaling2
    text_feature = scaling3
    flavor_feature = scaling4
    weights = []
    if(os.path.exists(f'{place}\\static\\images\\{cached_value}.svg')):
        return
    else:      
        check_if_not_0 = True
        if(color_feature > -1 ):
            file = open(f'{place}\\static\\cleaned_data\\colors_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0  = False
        if(check_if_not_0):
            used.append('color')
            dict_large['colors'] = list_to_add
            weights.append(color_feature)
        check_if_not_0 = True
        if(type_feature > -1 ):
            file = open(f'{place}\\static\\cleaned_data\\type_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0  = False
        if(check_if_not_0):
            used.append('type')
            dict_large['type'] = list_to_add
            weights.append(type_feature)
        check_if_not_0 = True
        if(text_feature > -1 ):
            file = open(f'{place}\\static\\cleaned_data\\text_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0 = False
        if(check_if_not_0):
            used.append('text')
            dict_large['text'] = list_to_add
            weights.append(text_feature)
        check_if_not_0 = True
        if(flavor_feature > -1 ):
            file = open(f'{place}\\static\\cleaned_data\\flavor_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0 = False
        if(check_if_not_0):
            used.append('flavor')
            dict_large['flavor'] = list_to_add
            weights.append(flavor_feature)
        data = json.load(open(f'{place}\\static\\cards_object.json'))
        used = {}
        for key in data:
            check_format = False
            check_colors = True
            check_rarity = False
            check_used = False
            try:
                for each in data[key]['legalities']:
                        if each['format'] in list1 and each['legality'] == 'Legal':
                            check_format = True
            except:
                pass
            try:
                for each in data[key]['colors']:
                    if each not in list2:
                        check_colors = False
            except:
                if('colorless' not in list2):
                    check_colors = True
            if data[key]['rarity'] in list3:
                check_rarity = True
            try:
                used[key] +=1
                check_used = True
            except:
                used[key] = 0
            if(check_format and check_colors and check_rarity and not check_used):
                list_keys.append(key)
                color = "grey"
                multi = 0
                try:    
                    for each in data[key]['colors']:
                        if each == 'W':
                            color = "white"
                            multi += 1
                        if each == 'U':
                            color = "blue"
                            multi += 1
                        if each == 'B':
                            color = "black"
                            multi += 1
                        if each == 'G':
                            color = "green"
                            multi += 1
                        if each == 'R':
                            color = "red"
                            multi += 1
                    if(multi) > 2:
                        color = "yellow"
                    elif(multi>1):
                        color = "yellow"
                except:
                    pass
                image = data[key]['image_url']
                list_vector.append(key)
        for nested_dict in dict_large:
            dict_large[nested_dict] = {key: dict_large[nested_dict][key] for key in list_keys}
        list_of_combined = []
        for key in list_keys:
            list_numpys = []
            for count,dict in enumerate(dict_large):
                value = dict_large[dict][key]
                value = value * new_scaling[count]
                list_numpys.append(value)
            combined = np.vstack(list_numpys)
            list_of_combined.append(combined)
        final_vectors_to_use = np.stack(list_of_combined)
            
        #vectors = _return_vectors_(scaling1,scaling2,scaling3,scaling4,cleaned_data)
        length_to_save = 5
        #Create_graph(vectors[0], 4,'Angel of Destiny', cached_value, length_to_save)
        smaller = nn_me(list_vector,final_vectors_to_use,weights,cached_value)
        final_dict = {}
        for count,key in enumerate(list_keys):
            final_dict[key] = []
            for j in range(final_vectors_to_use.shape[1]):
                final_dict[key].append(final_vectors_to_use[count,j,:])
            final_dict[key].append(final_vectors_to_use[count,:,:])
            final_dict[key].append(smaller[count])
        list_stuff = [final_vectors_to_use,list_keys,smaller,final_dict,used,loss_list]
        with open(f"static\\test_user_input\\modified_version2_{cached_value}_list.pickle", "wb") as fp:
            pickle.dump(list_stuff, fp)
        # for count,key in enumerate(vectors[0].keys()):
        #     arr = smaller[count]
        #     big_array = vectors[0][key]
        #     big_array.append(arr)
        #     vectors[0][key] = big_array
        # list_stuff = [cleaned_data,list_keys,vectors[0]]
        # with open(f"{cached_value}_list", "wb") as fp:
        #     pickle.dump(list_stuff, fp)

def calculate_sums(pickle_file):
    list_sums = []
    file = open(pickle_file,'rb')
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_to_add[0]):
        difference  = spatial.distance.cosine(each[0],list_after[count][0])
        sum = sum + difference
    list_sums.append(sum)
    print(f'color{str(x)} is '+ str(sum))
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        #print(each[0])
        #print(list_after[0])
        difference  = spatial.distance.cosine(each[1],list_after[count][1])
        sum = sum + difference
    list_sums.append(sum)
    print(f'type{str(x)} is '+ str(sum))
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        #print(each[0])
        #print(list_after[0])
        #changed = np.abs(each[2] - list_after[count][2])
        difference  = spatial.distance.cosine(each[2],list_after[count][2])
        sum = sum + difference
    list_sums.append(sum)
    print(f'text{str(x)} is '+ str(sum))
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        #print(each[0])
        #print(list_after[0])
        #changed = np.abs(each[3] - list_after[count][3])
        difference  = spatial.distance.cosine(each[3],list_after[count][3])
        sum = sum + difference
    list_sums.append(sum)
    print(f'flavor{str(x)} is '+ str(sum))
    return(list_sums)

def calculate_weights(list_sums,weights):
    max_sum = max(list_sums)
    for count,sum in enumerate(list_sums):
        list_sums[count] = max_sum/sum
    for count,value in enumerate(weights):
        weights[count] = list_sums[count] * value
    return(weights)

        

list1 = ['Standard','Pioneer']
list2 = ['Blue','White','Green','Black','Red']
list3 = ['Common','Uncommon','Rare','Mythic']
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
list_values = [0,0.2,0.4,0.6,0.8,1]
os.chdir(current_dir)
# for y in range(4):
#     value1 = random.random()
#     value2 = random.random()
#     value3 = random.random()
#     print(value1)
#     print(value2)
#     print(value3)
#     for x in list_values:
#         if y == 0:
#            _cluster_this(x,value1,value2,value3,list1,list2,list3,f'color{x},ty{value1},te{value2},fl{value3}')
#             _cluster_this(x,value1,value2,value3,list1,list2,list3,f'color{x}')
#         if y == 1:
#            _cluster_this(value1,x,value2,value3,list1,list2,list3,f'type{x},co{value1},te{value2},fl{value3}')
#             _cluster_this(x,value1,value2,value3,list1,list2,list3,f'type{x}')
#         if y == 2:
#            _cluster_this(value1,value2,x,value3,list1,list2,list3,f'text{x},co{value1},ty{value2},fl{value3}')
#             _cluster_this(x,value1,value2,value3,list1,list2,list3,f'text{x}')
#         if y == 3:
#            _cluster_this(value1,value2,value3,x,list1,list2,list3,f'flavor{x},co{value1},ty{value2},te{value3}')
#             _cluster_this(x,value1,value2,value3,list1,list2,list3,f'flavor{x}')
place = os.getcwd()
file = open(f'{place}\\static\\both_scores.pickle','rb')
modified_list = pickle.load(file)[1]
file.close()
for a in list_values:
    for b in list_values:
        for c in list_values:
            for d in list_values:
                if a != 0 or b!= 0 or c !=0 or d!= 0:
                    _cluster_this(modified_list,a,b,c,d,list1,list2,list3,f'co{a},ty{b},te{c},fl{d}')
                    
place = os.getcwd()
for x in list_values:
    file = open(f'{place}\\static\\test_user_input\\color_{x}_list.pickle','rb')
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_to_add[0]):
        difference  = spatial.distance.cosine(each[0],list_after[count][0])
        sum = sum + difference
    print(f'color{str(x)} is '+ str(sum))
    file = open(f'{place}\\static\\test_user_input\\type{x}_list.pickle','rb')
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        difference  = spatial.distance.cosine(each[1],list_after[count][1])
        sum = sum + difference
    print(f'type{str(x)} is '+ str(sum)) 
    file = open(f'{place}\\static\\test_user_input\\text{x}_list.pickle','rb')
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        difference  = spatial.distance.cosine(each[2],list_after[count][2])
        sum = sum + difference
    print(f'text{str(x)} is '+ str(sum))
    file = open(f'{place}\\static\\test_user_input\\flavor{x}_list.pickle','rb')
    list_to_add = pickle.load(file)
    list_before = list_to_add[0]
    list_after = list_to_add[2]
    sum = 0.0
    for count,each in enumerate(list_before):
        difference  = spatial.distance.cosine(each[3],list_after[count][3])
        sum = sum + difference
    print(f'flavor{str(x)} is '+ str(sum))
        
        
        