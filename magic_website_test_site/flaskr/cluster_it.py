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


stop_words = set(stopwords.words('english'))
stemmer = WordNetLemmatizer()
tokenizer = RegexpTokenizer(r'[\w\']+')
directory = os.getcwd()

def _get_text_stuff(all_data,dict_text):
    list_of_common_expressions = []
    window = 2
    key_amount = {}
    word_amount = {}
    type_amount = {}
    #Biword list
    for key in all_data:
        try:
            modified_string = re.sub(r"\([^()]*\)", "", all_data[key]['text'])
            tokenized = tokenizer.tokenize(modified_string)
            text_list = tokenized
            text_list = [w.lower() for w in text_list if not w.lower() in stop_words]
            text_list = [stemmer.lemmatize(word) for word in text_list]
            name_list= tokenizer.tokenize(all_data[key]['name'])
            name_list = [w.lower() for w in name_list if not w.lower() in stop_words]
            name_list = [stemmer.lemmatize(word) for word in name_list]
            for i, word in enumerate(text_list):
                for w in range(window):
                    if(i + 1 + w < len(text_list)):
                        if(text_list[(i + 1 + w)]) in name_list:
                            text_list[(i + 1 + w)] = "self"
                        if(word in name_list):
                            word =  "self"
                        try: 
                            key_amount[word + " " + text_list[(i + 1 + w)]] += 1
                        except:
                            key_amount[word + " " + text_list[(i + 1 + w)]] = 1
                if(word in name_list):
                    word =  "self"
                try: 
                    word_amount[word] += 1
                except:
                    word_amount[word] = 1
                        # Getting the context that is behind by *window* words    
                    #if i - w - 1 >= 0:
                    # try:
                        #   key_amount[word + " " + text_list[(i - 1 - w)]] += 1
                        #except: 
                            #key_amount[word + " " + text_list[(i - 1 - w)]] = 1
        except:
            pass
    sorted_x = sorted(key_amount.items(), reverse = True, key=lambda kv: kv[1])
    sorted_word = sorted(word_amount.items(), reverse = True, key=lambda kv: kv[1])
    for key in dict_text:
        count = 0
        for each in dict_text[key]:
            count = count + 1
        type_amount[key] = count
    sorted_type =  sorted(type_amount.items(), reverse = True, key=lambda kv: kv[1])
    placement = {}
    placement_word = {}
    placement_type = {}
    for count, key in enumerate(sorted_x): 
        placement[sorted_x[count][0]] = count
    for count, key in enumerate(sorted_type):
        placement_type[sorted_type[count][0]] = count
    for count, key in enumerate(sorted_word):
        placement_word[sorted_word[count][0]] = count
    return([sorted_x,sorted_word,placement,placement_word,placement_type,word_amount,key_amount,type_amount,window])

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

def create_weighted_loss(weights_input):
    def weighted_loss(y_true, y_pred):
        length = 0 
        weights = weights_input
        print("Shape of y_true: ", y_true.shape)
        print("Shape of y_pred: ", y_pred.shape)
        list_scaled_loss = []
        loss_total = 0
        for count,each in enumerate(weights_input):
            length = length + 1
            loss = tf.reduce_mean(tf.abs(y_true[count] - y_pred[count]))
            scaled_loss = loss * each
            loss_total = scaled_loss + loss_total
        # Return the mean of the weighted errors
        return (loss_total)
    
    return weighted_loss


# def create_weighted_loss(weights_input):
#     def weighted_loss(y_true, y_pred):
#         # Convert weights_input to a constant tensor
#         weights = K.constant(weights_input)

#         # Calculate absolute error
#         abs_error = K.abs(y_true - y_pred)

#         # Apply weights to the error
#         # Reshape weights to match the shape of abs_error
#         weights = K.reshape(weights, (1,len(weights_input), 1)) # Because you are weighting across the second dimension (4x100)
        
#         dynamic_shape = tf.shape(abs_error)
        
#         # Broadcast weights across batches and the 100 vector dimension
#         weights_broadcast = K.repeat_elements(weights, rep=dynamic_shape[0], axis=0) 
#         weights_broadcast = K.repeat_elements(weights_broadcast, rep=dynamic_shape.shape[2], axis=2) 

#         # Weight the errors
#         # This multiplies each value in abs_error by the corresponding weight
#         weighted_error = abs_error * weights_broadcast

#         # Calculate the mean of the weighted errors
#         weighted_mae = K.mean(weighted_error, axis=[1, 2]) # Take the mean across the last two dimensions

#         # Return the mean of the weighted errors
#         return K.mean(weighted_mae)
    
#     return weighted_loss

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
    hidden_encoder_layer = autoencoder.get_layer(name=encoded_layer).output
    hidden_second_layer = autoencoder.get_layer(name=second_layer).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden_encoder_layer)
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
    autoencoder.save(f"{model_hash}_my_model")
    autoencoder.save_weights(f"{model_hash}_my_model_weights")
    input_shape = autoencoder.input_shape
    X_encoded = encoder.predict(X_CNN)
    num_samples = X_CNN.shape[0]
    print('this_encoded')
    auto_encoded_text_vector = {}
    return(X_encoded)

def getImage(path):
   return OffsetImage(plt.imread(path, format="jpg"), zoom=.1)

def add_logo(background,foreground):
    bg_w, bg_h = background.size
    img_w, img_h = foreground.size
    img_offset = (20, (bg_h - img_h) // 4)
    background.paste(foreground, img_offset, foreground)
    return background

def add_color(image,c,transparency):
    color = Image.new('RGB',image.size,c)
    mask = Image.new('RGBA',image.size,(0,0,0,transparency))
    return Image.composite(image,color,mask).convert('RGB')

def write_image(background,color,text1,text2,foreground=''):
    background = add_color(background,color['c'],25)
    if not foreground:
        add_text(background,color,text1,text2)
    else:
        add_text(background,color,text1,text2,logo=True)
        add_logo(background,foreground)
    return background

def center_text(img,font,text1,text2,fill1,fill2):
    draw = ImageDraw.Draw(img) # Initialize drawing on the image
    w,h = img.size # get width and height of image
    t1_width, t1_height = draw.textsize(text1, font) # Get text1 size
    t2_width, t2_height = draw.textsize(text2, font) # Get text2 size
    p1 = ((w-t1_width)//7,h // 16) # H-center align text1
    p2 = ((w-t2_width)//7,h // 3 + h // 5) # H-center align text2
    draw.text(p1, text1, fill=fill1, font=font) # draw text on top of image
    draw.text(p2, text2, fill=fill2, font=font) # draw text on top of image
    return img

def add_text(img,color,text1,text2,logo=False,font='Roboto-Bold.ttf',font_size=500):
    draw = ImageDraw.Draw(img)
    p_font = color['p_font']
    s_font = color['s_font']
     
    # starting position of the message
    img_w, img_h = img.size
    height = img_h // 15.5
    font = ImageFont.load_default()
    #font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 28, encoding="unic")
 
    if logo == False:
        center_text(img,font,text1,text2,p_font,s_font)
    else:
        text1_offset = (img_w // 10, height)
        text2_offset = (img_w // 10, 5*height)
        draw.text(text1_offset, text1, fill=p_font, font=font)
        draw.text(text2_offset, text2, fill=s_font, font=font)
    return img
    
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
    X_encoded = NN_Put_Together(X,weights,saved_name,batches,epochs,filters,kernal)
    X_reduced = learn_manifold(X_encoded, umap_neighbors=30, umap_dim=int(encoded_dimensions/2))
    reducer = umap.UMAP(n_components=2)
    embedding = reducer.fit_transform(X_encoded)
    list_X = []
    list_Y = []
    for each in embedding:
        list_X.append(each[0])
        list_Y.append(each[1])
    smaller = X_encoded
    fig, ax = plt.subplots()
    fig.set_size_inches(50, 50)
    #For random values
    ax.scatter(list_X, list_Y)
    count = 0
    check = True
    dir = os.getcwd()+'\\static\\images\\card_images'
    current = os.getcwd()
    for count,name in enumerate(annotations):
        try:
            ab = AnnotationBbox(getImage(f"{dir}\\{name}.jpeg"), (list_X[count],list_Y[count]),frameon = False)
            ax.add_artist(ab)
        except:
            continue
    fig.savefig(f'static\\images\\{saved_name}.svg', format='svg', dpi=1200)
    return(smaller)

def _find_distances(list_cards,card1,card2):
    closest = 1
    saved = []
    len_to_save = 5
    VTC = list_cards[card1]
    VTC2 = list_cards[card2]
    list_distances = []
    for count, each in enumerate(VTC):
        list_distances.apend(spatial.distance.cosine(VTC[count],VTC2[count]))
    return(list_distances)

def _network_x_graph(list_cards,which_list,card1,len_to_save):
    saved = []
    VTC = list_cards[card1][which_list]
    check = False
    G = nx.empty_graph()
    G.add_node(card1)
    for count,name in enumerate(list_cards):
        if(name!= card1):
            value = spatial.distance.cosine(VTC,list_cards[name][which_list])
            if(len(saved)<len_to_save):
                saved.append([name,value])
            elif(check==False):
                saved.sort(key=lambda x:x[1], reverse = True)
                check = True
            if(len(saved)>=len_to_save):
                count2 = 0
                check2 = False
                while(value<saved[count2][1]):
                    count2 += 1
                    check2 = True
                    if(count2 == len_to_save):
                        break
                if(check2):
                    saved.insert(count2,[name,value])
                    saved.pop(0)
    list_cards_used = [card1] 
    for each in saved:
        G.add_node(each[0])
        G.add_edge(card1,each[0])
        list_cards_used.append(each[0])
    return([list_cards_used, G, saved])
    
def find_edges(use_list, key_card_name, len_to_save, which_list):
    list = use_list
    check = False
    for each in list:
        each.append([])
        each.append(check)
        each.append(check)
        each.append(key_card_name[each[0]][which_list])
        print(each[5])
    for count,name in enumerate(key_card_name):
        for each in list:
            if(name != each[0]):
                value = spatial.distance.cosine(each[5],key_card_name[name][which_list])
                if(len(each[2])<len_to_save):
                    each[2].append([name,value])
                elif(each[3]==False):
                    each[2].sort(key=lambda x:x[1], reverse = True)
                    each[3] = True
                if(len(each[2])>=len_to_save):
                    count2 = 0
                    each[4] = False
                    while(value<each[2][count2][1]):
                        count2 += 1
                        each[4] = True
                        if(count2 == len_to_save):
                            break
                    if(each[4]):
                        each[2].insert(count2,[name,value])
                        each[2].pop(0)
    return(list)

    #def write_edges(use_list):
    #   for each in use_list:           
    

def write_edges(G,list_cards_used,saved):
    for list in saved:
        for each in list[2]:
            G.add_node(each[0])
            G.add_edge(list[0],each[0])
            list_cards_used.append(each[0])
    return(G)
        #for each in list:
    
def Create_graph(list_cards, which_list, card1,cached_name, length):
    info = _network_x_graph(list_cards,which_list,card1,length)
    G = write_edges(info[1], info[0], find_edges(info[2],list_cards,length, which_list))
    plt.clf()
    nx.draw(info[1], with_labels = True)
    plt.savefig(f"static\\images\\{card1}{which_list}{cached_name}.png") 
    G.clear()
    plt.close()
    info[1].clear()
    info[0] = []
    info[2] = []
    
def _Compare_cards(vector,card1,card2,which_list):
    return(spatial.distance.cosine(vector[card1][which_list],vector[card2][which_list]))

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
            list_card_comparison[count].append(0)
    list_card_comparison =  sorted(list_card_comparison, reverse = False, key=lambda x: x[1])
    list_card_comparison.pop()
    return(list_card_comparison)

            
def _cluster_this(scaling1,scaling2,scaling3,scaling4,list1,list2,list3,cached_value):
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
    amount_of_blanks_to_add = 0
    if(os.path.exists(f'{place}\\static\\images\\{cached_value}.svg')):
        return
    else:      
        check_if_not_0 = True
        if(color_feature > 0 ):
            file = open(f'{place}\\static\\cleaned_data\\colors_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0  = False
            amount_of_blanks_to_add+1
        if(check_if_not_0):
            used.append('color')
            dict_large['colors'] = list_to_add
            weights.append(color_feature)
        check_if_not_0 = True
        if(type_feature > 0 ):
            file = open(f'{place}\\static\\cleaned_data\\type_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0  = False
            amount_of_blanks_to_add+1
        if(check_if_not_0):
            used.append('type')
            dict_large['type'] = list_to_add
            weights.append(type_feature)
        check_if_not_0 = True
        if(text_feature > 0 ):
            file = open(f'{place}\\static\\cleaned_data\\text_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0 = False
            amount_of_blanks_to_add+1
        if(check_if_not_0):
            used.append('text')
            dict_large['text'] = list_to_add
            weights.append(text_feature)
        check_if_not_0 = True
        if(flavor_feature > 0 ):
            file = open(f'{place}\\static\\cleaned_data\\flavor_numpy_dict.obj','rb')
            list_to_add = pickle.load(file)
            file.close()
        else:
            check_if_not_0 = False
            amount_of_blanks_to_add+1
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
            for dict in dict_large:
                value = dict_large[dict][key]
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
        list_stuff = [final_vectors_to_use,list_keys,smaller,final_dict,used]
        with open(f"static\\images\\{cached_value}_list", "wb") as fp:
            pickle.dump(list_stuff, fp)
        # for count,key in enumerate(vectors[0].keys()):
        #     arr = smaller[count]
        #     big_array = vectors[0][key]
        #     big_array.append(arr)
        #     vectors[0][key] = big_array
        # list_stuff = [cleaned_data,list_keys,vectors[0]]
        # with open(f"{cached_value}_list", "wb") as fp:
        #     pickle.dump(list_stuff, fp)