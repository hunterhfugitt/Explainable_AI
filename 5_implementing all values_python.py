from memory_profiler import profile
import math
import os
import signal
import sys
sys.setrecursionlimit(10**9)
import threading
threading.stack_size(10**8)
from tempfile import TemporaryDirectory
from typing import Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense, Input
from keras.models import Model
from datetime import datetime
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import keras
import sklearn
from scipy import spatial
from torchtext.data.utils import get_tokenizer
import numpy as np
import pickle
from keras.layers import Conv2D, Conv2DTranspose, Flatten, Reshape
from keras.layers import Input
from keras.models import Model
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device("cuda")
print(os.getcwd())
import pdb
import torch
from transformers import BertModel, BertForMaskedLM, BertTokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def signal_handler(sig, frame):
    print(f"Caught signal {sig}, exiting!")
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

#checkpoint = torch.load('C:\\Users\\hunte\\coding projects\\ipynbs to get good text data\\example.pt')
dir = os.getcwd
model_path = f'{dir}/example_fixed.pt'
print(device)
#tf.debugging.set_log_device_placement(True)
file = open("cleaned_data\\features.pickle",'rb')
feature_list = pickle.load(file)
file.close()
relevant_dicts = []

for feature in feature_list:
    relevant_dicts.append(json.load(open(f'cleaned_data\\cleaned_up_{feature}_data.json')))
    
    
class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

file = open("vocab.obj",'rb')
vocab = pickle.load(file)
file.close()
#input_text = text_data[text_data[0]]
print(type(vocab))
ntokens = len(vocab) # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.to('cuda')
#print(type(state_dict))
#print(type(model))

#for key, value in state_dict.items():
    #print(key, value)

#print(len(vocab))
#for x in range(0,300):
    #print(vocab.lookup_token(x))
new_lists = []
start_list = []
for count,feature_dict in enumerate(relevant_dicts):
    new_list = []
    for value in feature_dict:
        new_list.append(value)
    if count == 0:
        start_list.append(value)
    new_lists.append(new_list)


def get_conv_autoencoder(input_shape, filters, kernel_size, act='relu'):
    # Number of encoding and decoding layers
    n_stacks = len(filters) - 1

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

def NN_Put_Together(X,feature,batch,epochs,filters,kernal):

    X_train, X_test = train_test_split(X, test_size=0.2, random_state=11)
    batch_size = batch
    pretrain_epochs = epochs
    #shape = [X.shape[-1], 2500, 500, 1000, encoded_dimensions]
    shapeD1 = X.shape[-2]
    shapeD2 = X.shape[-1]
    #shape = [X.shape[-1], 1000, 1000, 1500, 250, 250, 500, encoded_dimensions]
    #shape = [X.shape[-1], 1000, 1000, 1500, 750, 750, 500, 250, 250, 500, encoded_dimensions]
    shape = (shapeD1,shapeD2,1)
    autoencoder = get_conv_autoencoder(shape,filters,kernal)
    print(f'this is for {feature}')
    print("this is the shape")
    print(shapeD1,shapeD2)
    encoded_layer = f'encoder_{len(filters) - 1}'
    second_layer = 'encoder_0'
    hidden_encoder_layer = autoencoder.get_layer(name=encoded_layer).output
    hidden_second_layer = autoencoder.get_layer(name=second_layer).output
    encoder = Model(inputs=autoencoder.input, outputs=hidden_encoder_layer)
    autoencoder.compile(loss='mse', optimizer='adam')
    #train the autoencoder
    model_series = f'CLS_MODEL_{feature}' + datetime.now().strftime("%h%d%Y-%H%M")

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
    autoencoder = load_model(f"{model_series}-model.h5")
    autoencoder.save(f"{feature}_my_model")
    autoencoder.save_weights(f"{feature}_my_model_weights")
    #print(X_CNN)
    input_shape = autoencoder.input_shape
    print("Input shape:", input_shape)
    print(X_CNN.shape)
    X_encoded = encoder.predict(X_CNN)
    #batch_size = 10  # Adjust this value as per your memory capacity
    num_samples = X_CNN.shape[0]
    print('this_encoded')
    auto_encoded_text_vector = {}
    print(len(X_encoded))
    del X,X_train,X_test,X_CNN,autoencoder, encoder
    with open(f"{feature}_numpy.obj", "wb") as fp:
        pickle.dump(X_encoded, fp)
    return(X_encoded)
@profile    
def _encode_me(model,feature,list_to_be_encoded,count):
    list_tensors = []
    tokenizer = get_tokenizer('basic_english')
    max = 0
    np_tensor_array = np.array([]).reshape(0, 1)
    for value in list_to_be_encoded:
        tokenized_text = tokenizer(relevant_dicts[count][value])
        input_ids = torch.tensor(vocab(tokenized_text), dtype=torch.long)
        input_ids = input_ids.to('cuda')
        encoder_layer = model.encoder(input_ids.int())
        #encoder_layer = model.encoder(input_ids)
        encoder_layer = encoder_layer.cpu().detach().numpy()
        list_tensors.append(encoder_layer)
        if(max < len(tokenized_text)):
            max = len(tokenized_text)
    padded_arrays = []
    for each in list_tensors:
        #print(len(each))
        #print(max)
        pad_length = max - len(each)
        padded_arr = np.pad(each, pad_width=((0, pad_length), (0, 0)), mode='constant')
        padded_arrays.append(padded_arr)
    padded_array = np.stack(padded_arrays)
    #padded_tensor = torch.nn.utils.rnn.pad_sequence(list_tensors, batch_first=True, padding_value=0)
    #padded_numpy = padded_tensor.cpu().detach().numpy()
    filters = [16,32,16,100]
    print(padded_array.shape)
    return(NN_Put_Together(padded_array,feature,256,4,filters,(3,3)))
    
ntokens = len(vocab) # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)
os.chdir(rf'{dir}\\cleaned_data')
for count,each in enumerate(new_lists):
   #if feature_list[count] == 'type':
    feature = feature_list[count]
    list_features_encoded = ''
    try:
        list_features_encoded = _encode_me(model,feature,each,count)
        print('we_got_here')
        auto_encoded_text_vector = {}
        auto_encoded_text_with_numpy = {}
        for count2,name in enumerate(each):
            auto_encoded_text_vector[name] = {}
            auto_encoded_text_vector[name]['full_vector'] = list_features_encoded[count2].tolist()
            auto_encoded_text_with_numpy[name] = {}
            auto_encoded_text_with_numpy[name] = list_features_encoded[count2]
        with open(f"encoded_{feature}_data.json", "w") as f:
            json.dump(auto_encoded_text_vector, f)
        with open(f"{feature}_numpy_dict.obj", "wb") as fp:
            pickle.dump(auto_encoded_text_with_numpy, fp)
    except:
        pass
    import shutil
    
    source_dir = os.getcwd()
    destination_dir = f'{dir}//magic_website_test_site//flaskr//static//cleaned_data'
    try:
        shutil.copytree(source_dir, destination_dir)
        print(f"Directory '{source_dir}' has been copied to '{destination_dir}' successfully.")
    except shutil.Error as e:
        print(f"Directory not copied. Error: {e}")
    except OSError as e:
        print(f"Directory not copied. OSError: {e}")