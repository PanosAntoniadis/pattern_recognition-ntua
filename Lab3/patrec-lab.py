#!/usr/bin/env python
# coding: utf-8

# # Classify Genres and Emotions in Songs Using Deep Learning

# ## Description:
# 
# The goal of this lab is to recognize the genre and extract the emotions from spectrograms of music songs. We are given 2 datasets:
# 
# - Free Music Archive (FMA) genre that contains 3834 samples from 20 music genres.
# - Multitask music dataset that contains 1497 samples with labels about the emotions such as valence, energy and danceability.
# 
# All samples came from spectrograms, that have been extracted from clips of 30 seconds from different songs. We will analyze the spectrograms using deep learning architectures such as Recurrent Neural Networks and Convolutional Neural Networks. The exercise is separated in 5 parts:
# 
# - Data analysis and familiarize with spectrograms.
# - Implement classifiers about the music genre using the FMA dataset.
# - Implement regression models for predicting valence, energy and danceability.
# - Use of modern training techniques, such as transfer and multitask learning, to improve the previous results.
# - Submit results in the Kaggle competition of the exercise (optional).

# In[1]:


# Import necessary libraries
import numpy as np
import copy
import re
import os
import pandas as pd
import random
import matplotlib.pyplot as plt

# Sklearn
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Pytorch
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models

# Scipy
from scipy.stats import spearmanr

# Other
from IPython.display import Image


# ## Data Loading

# In this section, all the code for loading and manipulating the 2 datasets is available. Some parts are the same with the prepare_lab.

# In[2]:


# Combine similar classes and remove underrepresented classes.

class_mapping = {
    'Rock': 'Rock',
    'Psych-Rock': 'Rock',
    'Indie-Rock': None,
    'Post-Rock': 'Rock',
    'Psych-Folk': 'Folk',
    'Folk': 'Folk',
    'Metal': 'Metal',
    'Punk': 'Metal',
    'Post-Punk': None,
    'Trip-Hop': 'Trip-Hop',
    'Pop': 'Pop',
    'Electronic': 'Electronic',
    'Hip-Hop': 'Hip-Hop',
    'Classical': 'Classical',
    'Blues': 'Blues',
    'Chiptune': 'Electronic',
    'Jazz': 'Jazz',
    'Soundtrack': None,
    'International': None,
    'Old-Time': None
}


# In[3]:


# Define a function that splits a dataset in train and validation set.

def torch_train_val_split(dataset, batch_train, batch_eval, val_size=.2, shuffle=True, seed=None):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    # Rearrange train and validation set
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_train,
                              sampler=train_sampler)
    val_loader = DataLoader(dataset,
                            batch_size=batch_eval,
                            sampler=val_sampler)
    return train_loader, val_loader


# In[4]:


# Define some useful functions for loading spectrograms and chromagrams

def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T


# In[5]:


# Define an encoder for the labels

class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


# In[6]:


# Define a PaddingTransformer in order to convert all input sequences to the same length

class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[:self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


# In[7]:


# Define useful parameters that are the same for all the models.

num_mel = 128
num_chroma = 12
n_classes = 10


# - Define a Pytorch dataset for the 1st dataset.

# In[8]:


class SpectrogramDataset(Dataset):
    def __init__(self, path, class_mapping=None, train=True, max_length=-1, read_spec_fn=read_fused_spectrogram):
        t = 'train' if train else 'test'
        p = os.path.join(path, t)
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(self.label_transformer.fit_transform(labels)).astype('int64')

    def get_files_labels(self, txt, class_mapping):
        with open(txt, 'r') as fd:
            lines = [l.rstrip().split('\t') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0].split('.')[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
            labels.append(label)
        return files, labels
    

    def __getitem__(self, item):
        # Return a tuple in the form (padded_feats, label, length)
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)


# - Load mel spectrograms.

# In[9]:


mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)

train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs, 32 ,32, val_size=.33)


# In[10]:


print("Shape of a train example in SpectrogramDataset: ")
print(mel_specs[1][0].shape)


# We should pad the test set, so as to have the same shape with the train dataset (in order to use them in the following CNN).

# In[11]:


test_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=1293,
         read_spec_fn=read_mel_spectrogram)

test_loader_mel = DataLoader(test_mel, batch_size=32)


# ### Step 7: 2D CNN
# 
# Antother way of contructing a model for analyzing speech singnals is to see the spectrogram as an image and use a Convolutional Neural Network.

# - [Here](https://cs.stanford.edu/people/karpathy/convnetjs/) we can train simple Convolutional Networks and check their internal functions by displaying the activations of each layer without programming cost. We will train a CNN in MNIST and comment on the functions.

# In[12]:


Image("../input/images/convnetjs_1.png")


# The dataset is fairly easy and the accuracy for the validation set is very high.

# In[13]:


Image("../input/images/convnetjs_2.png")


# In[14]:


Image("../input/images/convnetjs_3.png")


# In each layer, each neuron learns a part of the corresponding digit.

# In[15]:


Image("../input/images/convnetjs_4.png")


# The only wrong prediction on the test set is in an image that even a person may predict wrong since it is not clearly written.

# - Define a 2D CNN with 4 layers. Each layer implements the following operations:
# 
#     - 2D convclution: The purpose of this step is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data
#     
#     - Batch normalization: Speeds up learning.
#     
#     - ReLU activation: Using ReLU, we introduce non-linearity in our ConvNet, since most of the real-world data we would want our ConvNet to learn would be non-linear.
#     
#     - Max pooling:  Reduces the dimensionality of each feature map but retains the most important information. 

# In[16]:


# Define a function that trains the model for an epoch.

def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # Εnable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # Οbtain the model's device ID
    device = next(model.parameters()).device
    
    for index, batch in enumerate(dataloader, 1):
        # Get the inputs (batch)
        inputs, labels, lengths = batch

        # Move the batch tensors to the right device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad()

        # Step 2 - forward pass: y' = model(x)
        y_preds = model(inputs, lengths)
        
        # Step 3 - compute loss: L = loss_function(y, y')
        loss = loss_function(y_preds, labels)

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # Step 5 - update weights
        optimizer.step()

        # Accumulate loss in a variable.
        running_loss += loss.data.item()

    return running_loss / index


# In[17]:


# Define a function that evaluates the model in an epoch.
def eval_dataset(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # Disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # Obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()

    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # Get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Step 2 - forward pass: y' = model(x)
            y_preds = model(inputs, lengths)  # EX9

            # Step 3 - compute loss: L = loss_function(y, y')
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(y_preds, labels)

            # Step 4 - make predictions (class = argmax of posteriors)
            y_preds_arg = torch.argmax(y_preds, dim=1)

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(y_preds_arg.cpu().numpy())
            y.append(labels.cpu().numpy())

            # Accumulate loss in a variable
            running_loss += loss.data.item()

    return running_loss / index, (y, y_pred)


# In[18]:


# Define the CNN architecture

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv1_bn = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.conv2_bn = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(6, 8, 3)
        self.conv3_bn = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(8, 12, 3)
        self.conv4_bn = nn.BatchNorm2d(12)
        self.pool4 = nn.MaxPool2d(2, 2)
                
        self.fc1 = nn.Linear(4680, 128)
        
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, lengths):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        
        x = self.pool1(F.relu( self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu( self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu( self.conv3_bn(self.conv3(x))))
        x = self.pool4(F.relu( self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Train the model in the mel spectrograms.

# In[19]:


EPOCHS = 15
model = ConvNet()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
n_epochs_stop = 4
min_val_loss = 1000
epochs_no_improve = 0

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_mel, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_mel, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_mel, model, loss_function)
    
    if val_loss < min_val_loss:
        # Save the model
        torch.save(model, "./mel_cnn")
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
            
    if epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break
        
    y_train_true = np.concatenate( y_train_gold, axis=0 )
    y_val_true = np.concatenate( y_val_gold, axis=0 )
    y_train_pred = np.concatenate( y_train_pred, axis=0 )
    y_val_pred = np.concatenate( y_val_pred, axis=0 )
    print("Epoch %d " %epoch)
    print("Train loss: %f" %train_loss)
    print("Validation loss: %f" %val_loss)
    print("Accuracy for train:" , accuracy_score(y_train_true, y_train_pred))
    print("Accuracy for validation:" , accuracy_score(y_val_true, y_val_pred))
    print()
    
# Save the model for future evaluation  
torch.save(model, './mel_cnn')


# In[20]:


# Load the model
model = torch.load('./mel_cnn')
model.eval()


# Evaluate the cnn model in the test set

# In[21]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_mel, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Test loss: %f" %test_loss)
print("Accuracy for test:" , accuracy_score(y_test_true, y_test_pred))
print()
    


# We observe that the CNN architecture increased the accuracy of our model.

# ## Step 8: Sentiment prediction with regression

# Now, we will use the multitask dataset.

# In[22]:


# Define a Pytorch dataset for the Multitask dataset

class MultitaskDataset(Dataset):
    def __init__(self, path, max_length=-1, read_spec_fn=read_fused_spectrogram, label_type='energy'):
        p = os.path.join(path, 'train')
        self.label_type = label_type
        self.index = os.path.join(path, "train_labels.txt")
        self.files, labels = self.get_files_labels(self.index)
        self.feats = [read_spec_fn(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length) 
        if isinstance(labels, (list, tuple)):
            self.labels = np.array(labels)

    def get_files_labels(self, txt):
        with open(txt, 'r') as fd:
            lines = [l.split(',') for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.label_type == 'valence':
                labels.append(float(l[1]))
            elif self.label_type == 'energy':
                labels.append(float(l[2]))
            elif self.label_type == 'danceability':
                labels.append(float(l[3].strip("\n")))
            else:
                labels.append([float(l[1]), float(l[2]), float(l[3].strip("\n"))])
            # Kaggle automatically unzips the npy.gz format so this hack is needed
            _id = l[0]
            npy_file = '{}.fused.full.npy'.format(_id)
            files.append(npy_file)
        return files, labels
    

    def __getitem__(self, item):
        # Return a tuple in the form (padded_feats, valence, energy, danceability, length)
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], l

    def __len__(self):
        return len(self.labels)


# - Load mulstitask dataset with certain padding in order to fit in the previous CNN.
# 

# In[23]:


specs_multi_valence = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         label_type='valence',
         read_spec_fn=read_mel_spectrogram)

train_loader_valence , val_loader_valence = torch_train_val_split(specs_multi_valence, 32 ,32, val_size=.33)


# In[24]:


specs_multi_energy = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         label_type='energy',
         read_spec_fn=read_mel_spectrogram)

train_loader_energy, val_loader_energy = torch_train_val_split(specs_multi_energy, 32 ,32, val_size=.33)


# In[25]:


specs_multi_danceability = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         label_type='danceability',
         read_spec_fn=read_mel_spectrogram)

train_loader_danceability, val_loader_danceability = torch_train_val_split(specs_multi_danceability, 32 ,32, val_size=.33)


# In[26]:


print(specs_multi_valence[0][0].shape)
print(specs_multi_energy[0][0].shape)
print(specs_multi_danceability[0][0].shape)


# - Load beat mulstitask dataset for the LSTM.
# 

# In[27]:


beat_specs_multi_valence = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         max_length=-1,
         label_type='valence',
         read_spec_fn=read_mel_spectrogram)

beat_train_loader_valence , beat_val_loader_valence = torch_train_val_split(beat_specs_multi_valence, 32 ,32, val_size=.33)


# In[28]:


beat_specs_multi_energy = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         max_length=-1,
         label_type='energy',
         read_spec_fn=read_mel_spectrogram)

beat_train_loader_energy, beat_val_loader_energy = torch_train_val_split(beat_specs_multi_energy, 32 ,32, val_size=.33)


# In[29]:


beat_specs_multi_danceability = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset_beat/',
         max_length=-1,
         label_type='danceability',
         read_spec_fn=read_mel_spectrogram)

beat_train_loader_danceability, beat_val_loader_danceability = torch_train_val_split(beat_specs_multi_danceability, 32 ,32, val_size=.33)


# In[30]:


print(beat_specs_multi_valence[0][0].shape)
print(beat_specs_multi_energy[0][0].shape)
print(beat_specs_multi_danceability[0][0].shape)


# - Define the LSTM of Step 5 (prepare_lab) and train the beat mel multitask dataset.

# In[31]:


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.rnn_size = rnn_size
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize the LSTM, Dropout, Output layers
        
        self.lstm = nn.LSTM(input_dim, self.rnn_size, self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # Obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        
        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.rnn_size).double().to(DEVICE)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.rnn_size).double().to(DEVICE)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size).double().to(DEVICE)
            c0 = torch.zeros(self.num_layers, x.size(0), self.rnn_size).double().to(DEVICE)
            
        # Forward propagate LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Forward propagate Linear
        last_outputs = self.linear(self.last_timestep(lstm_out, lengths, self.bidirectional))
        return last_outputs.view(-1)

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Obtain the model's device ID
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1).to(DEVICE)
        return outputs.gather(1, idx).squeeze()


# In[32]:


# Define train function for regression.

def train_dataset_regression(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # Εnable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # Οbtain the model's device ID
    device = next(model.parameters()).device
    
    for index, batch in enumerate(dataloader, 1):
        # Get the inputs (batch)
        inputs, labels, lengths = batch

        # Move the batch tensors to the right device
        inputs = inputs.to(device)
        labels = labels.to(device)
            
    
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad()

        # Step 2 - forward pass: y' = model(x)
        y_preds = model(inputs, lengths)
                
        # Step 3 - compute loss: L = loss_function(y, y')
        loss = loss_function(y_preds, labels.double())

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # Step 5 - update weights
        optimizer.step()

        # Accumulate loss in a variable.
        running_loss += loss.data.item()

    return running_loss / index


# In[33]:


# Define evaluation function for regression.

def eval_dataset_regression(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # Disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # Obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()

    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # Get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Step 2 - forward pass: y' = model(x)
            y_preds = model(inputs, lengths)  # EX9

            # Step 3 - compute loss: L = loss_function(y, y')
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(y_preds, labels.double())

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(y_preds.cpu().numpy())
            y.append(labels.cpu().numpy())

            # Accumulate loss in a variable
            running_loss += loss.data.item()

    return running_loss / index, (y, y_pred)


# In order to train the 2nd dataset in the models that we have defined, we should change the loss function for regression. We will use mean squared error.

# - Training for valence in LSTM.

# In[34]:


RNN_SIZE = 32
EPOCHS = 20

model = BasicLSTM(num_mel, RNN_SIZE, 1, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, beat_train_loader_valence, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, _ = eval_dataset_regression(beat_train_loader_valence, model, loss_function)

    val_loss, _ = eval_dataset_regression(beat_val_loader_valence, model, loss_function)
    
    if epoch%(5) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        print()
        
torch.save(model, './multitask_lstm_valence')


# In[35]:


model = torch.load('./multitask_lstm_valence')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(beat_val_loader_valence, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# - Training for energy in LSTM.

# In[36]:


RNN_SIZE = 32
EPOCHS = 20

model = BasicLSTM(num_mel, RNN_SIZE, 1, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, beat_train_loader_energy, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, _ = eval_dataset_regression(beat_train_loader_energy, model, loss_function)

    val_loss, _ = eval_dataset_regression(beat_val_loader_energy, model, loss_function)
    
    if epoch%(5) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        print()
        
torch.save(model, './multitask_lstm_energy')


# In[37]:


model = torch.load('./multitask_lstm_energy')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(beat_val_loader_energy, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# - Training for danceability in LSTM.

# In[38]:


RNN_SIZE = 32
EPOCHS = 20

model = BasicLSTM(num_mel, RNN_SIZE, 1, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, beat_train_loader_danceability, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, _ = eval_dataset_regression(beat_train_loader_danceability, model, loss_function)

    val_loss, _ = eval_dataset_regression(beat_val_loader_danceability, model, loss_function)
    
    if epoch%(5) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        print()
        
torch.save(model, './multitask_lstm_danceability')


# In[39]:


model = torch.load('./multitask_lstm_danceability')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(beat_val_loader_danceability, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# - Define the CNN of Step 7 and train the mel multitask dataset.

# In[40]:


# Define the CNN architecture

class ConvNetMultitask(nn.Module):
    def __init__(self):
        super(ConvNetMultitask, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.conv1_bn = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(3, 6, 5)
        self.conv2_bn = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(6, 8, 3)
        self.conv3_bn = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(8, 12, 3)
        self.conv4_bn = nn.BatchNorm2d(12)
        self.pool4 = nn.MaxPool2d(2, 2)
                
        self.fc1 = nn.Linear(4680, 128)
        
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, lengths):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        
        x = self.pool1(F.relu( self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu( self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu( self.conv3_bn(self.conv3(x))))
        x = self.pool4(F.relu( self.conv4_bn(self.conv4(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# - Training for valence in CNN.

# In[41]:


EPOCHS = 20

model = ConvNetMultitask()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_valence, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_valence, model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_valence, model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()
        
torch.save(model, './multitask_cnn_valence')


# In[42]:


model = torch.load('./multitask_cnn_valence')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_valence, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# - Training for energy in CNN.

# In[43]:


EPOCHS = 20

model = ConvNetMultitask()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_energy, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_energy, model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_energy, model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()
        
torch.save(model, './multitask_cnn_energy')


# In[44]:


model = torch.load('./multitask_cnn_energy')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_energy, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# - Training for danceability in CNN.

# In[45]:


EPOCHS = 20

model = ConvNetMultitask()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_danceability, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_danceability, model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_danceability, model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()
        
torch.save(model, './multitask_cnn_danceability')


# In[46]:


model = torch.load('./multitask_cnn_danceability')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_danceability, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# ## Step 9a: Transfer Learning

# When we have little available data, we can increase the performance of our deep neural networks using transfer learning from another model, trained on a bigger dataset.

# - We choose the CNN architecture of step 7. The idea of Transfer Learning came in to picture when researchers realized that the first few layers of a CNN are learning low-level features like edges and corners. So, there is no point learning the same thing again while training on a similar data. But, we don’t have a clear-cut intuition on what is learned at different layers in a LSTM or GRU, since it is a time series model, making the whole process very complex.

# - We load the model that is already trained in fma_genre_spectrograms.

# In[47]:


transfer_model = torch.load('./mel_cnn')
print(transfer_model)


# In[48]:


# We freeze the parameters
for param in transfer_model.parameters():
    param.requires_grad = False


# In[49]:


# We change only the last layer to fit in our new regression problem.
transfer_model.fc2 = nn.Linear(128, 1)
print(transfer_model)


# Now, we will train only the last layer of our freeze model in the regression problem.

# In[51]:


EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model.double()
transfer_model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_valence, transfer_model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_valence, transfer_model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_valence, transfer_model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()

        
torch.save(transfer_model, './multitask_transfer_valence')


# In[52]:


transfer_model = torch.load('./multitask_transfer_valence')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_valence, transfer_model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# In[53]:


transfer_model = torch.load('./mel_cnn')

# We freeze the parameters
for param in transfer_model.parameters():
    param.requires_grad = False
    
# We change only the last layer to fit in our new regression problem.
transfer_model.fc2 = nn.Linear(128, 1)


EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model.double()
transfer_model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_energy, transfer_model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_energy, transfer_model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_energy, transfer_model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()

        
torch.save(transfer_model, './multitask_transfer_energy')


# In[54]:


transfer_model = torch.load('./multitask_transfer_energy')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_energy, transfer_model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# In[55]:


transfer_model = torch.load('./mel_cnn')

# We freeze the parameters
for param in transfer_model.parameters():
    param.requires_grad = False
    
# We change only the last layer to fit in our new regression problem.
transfer_model.fc2 = nn.Linear(128, 1)


EPOCHS = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transfer_model.double()
transfer_model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(transfer_model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression(epoch, train_loader_danceability, transfer_model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression(train_loader_danceability, transfer_model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression(val_loader_danceability, transfer_model, loss_function)
    
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )
        print("Spearman in train: %f" %spearmanr(y_train_true, y_train_pred)[0])
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation: %f" %spearmanr(y_val_true, y_val_pred)[0])
        print()

        
torch.save(transfer_model, './multitask_transfer_danceability')


# In[56]:


transfer_model = torch.load('./multitask_transfer_danceability')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression(val_loader_danceability, transfer_model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )
print("Spearman: %f" %spearmanr(y_test_true, y_test_pred)[0])
print()


# ## Step 9b: Multitask Learning

# In step 8, we trained a separate model for each emotion. Another way of training more efficient models when we have many labels is to use multitask learning.

# Load dataset that contains all three labels

# In[57]:


specs_multi_all = MultitaskDataset(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         label_type=-1,
         read_spec_fn=read_mel_spectrogram)

train_loader_all, val_loader_all = torch_train_val_split(specs_multi_all, 32 ,32, val_size=.33)


# In[58]:


print("Shape of an example: ")
print(specs_multi_all[0][0].shape)
print("Shape of label: ")
print(specs_multi_all[0][1].shape)


# In[59]:


# Define trainning function for regression in multitask learning.

def train_dataset_regression_multi(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # Εnable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # Οbtain the model's device ID
    device = next(model.parameters()).device
    
    for index, batch in enumerate(dataloader, 1):
        # Get the inputs (batch)
        inputs, labels, lengths = batch

        # Move the batch tensors to the right device
        inputs = inputs.to(device)
        labels = labels.to(device)
    
        
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad()

        # Step 2 - forward pass: y' = model(x)
        y_preds_val, y_preds_energy, y_preds_dance = model(inputs, lengths)
        
        # Step 3 - compute loss: L = loss_function(y, y')
        loss_1 = loss_function(y_preds_val, labels[:, 0].double())
        loss_2 = loss_function(y_preds_energy, labels[:, 1].double())
        loss_3 = loss_function(y_preds_dance, labels[:, 2].double())
        loss = loss_1 + loss_2 + loss_3

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # Step 5 - update weights
        optimizer.step()

        # Accumulate loss in a variable.
        running_loss += loss.data.item()

    return running_loss / index


# In[60]:


# Define evaluation function for regression multitask learning.

def eval_dataset_regression_multi(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # Disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []
    
    y = []


    # Obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()

    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # Get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Step 2 - forward pass: y' = model(x)
            y_preds_val, y_preds_energy, y_preds_dance = model(inputs, lengths)

            # Step 3 - compute loss: L = loss_function(y, y')
            loss_1 = loss_function(y_preds_val, labels[:, 0].double())
            loss_2 = loss_function(y_preds_energy, labels[:, 1].double())
            loss_3 = loss_function(y_preds_dance, labels[:, 2].double())
        
            loss = loss_1 + loss_2 + loss_3
        
            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(np.hstack((y_preds_val.cpu().numpy(), y_preds_energy.cpu().numpy(), y_preds_dance.cpu().numpy())))
            y.append(labels.cpu().numpy())

            # Accumulate loss in a variable
            running_loss += loss.data.item()
    return running_loss / index, (y, y_pred)


# In[61]:


class ConvNetMultitaskLearning(nn.Module):
    def __init__(self):
        super(ConvNetMultitaskLearning, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3)
        self.conv1_bn = nn.BatchNorm2d(3)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv2_bn = nn.BatchNorm2d(6)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(6, 8, 3)
        self.conv3_bn = nn.BatchNorm2d(8)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.conv4 = nn.Conv2d(8, 12, 3)
        self.conv4_bn = nn.BatchNorm2d(12)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.conv5 = nn.Conv2d(12, 16, 3)
        self.conv5_bn = nn.BatchNorm2d(16)
        self.pool5 = nn.MaxPool2d(2, 2)
                
        self.fc1 = nn.Linear(1216, 32)
        
        self.fc_val = nn.Linear(32, 1)
        
        self.fc_energy = nn.Linear(32, 1)
        
        self.fc_dance = nn.Linear(32, 1)

    def forward(self, x, lengths):
        x = x.view(x.size(0), 1, x.size(1), x.size(2))
        
        x = self.pool1(F.relu( self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu( self.conv2_bn(self.conv2(x))))
        x = self.pool3(F.relu( self.conv3_bn(self.conv3(x))))
        x = self.pool4(F.relu( self.conv4_bn(self.conv4(x))))
        x = self.pool5(F.relu( self.conv5_bn(self.conv5(x))))
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        energy = self.fc_energy(x)
        
        val = self.fc_val(x)
        
        dance = self.fc_dance(x)
        
        return val, energy, dance


# In[62]:


EPOCHS = 50

model = ConvNetMultitaskLearning()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
n_epochs_stop = 10
min_val_loss = 1000
epochs_no_improve = 0

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset_regression_multi(epoch, train_loader_all, model, loss_function, optimizer)
    
    #train_dataset_regression_multi(epoch, val_loader_all, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train, y_train_pred) = eval_dataset_regression_multi(train_loader_all, model, loss_function)

    val_loss, (y_val, y_val_pred) = eval_dataset_regression_multi(val_loader_all, model, loss_function)
    
    if val_loss < min_val_loss:
        # Save the model
        torch.save(model, './multitask_learning')
        epochs_no_improve = 0
        min_val_loss = val_loss
    else:
        epochs_no_improve += 1
            
    if epochs_no_improve == n_epochs_stop:
        print('Early stopping!')
        break
    if epoch%(1) == 0: 
        print("Epoch %d " %epoch)
        print("Train loss: %f" %train_loss)
        print("Validation loss: %f" %val_loss)
        y_train_true = np.concatenate( y_train, axis=0 )
        y_train_pred = np.concatenate( y_train_pred, axis=0 )

        print("Spearman in train - valence: %f" %spearmanr(y_train_true[:,0], y_train_pred[:,0])[0])
        print("Spearman in train - energy: %f" %spearmanr(y_train_true[:,1], y_train_pred[:,1])[0])
        print("Spearman in train - dance: %f" %spearmanr(y_train_true[:,2], y_train_pred[:,2])[0])
        
        y_val_true = np.concatenate( y_val, axis=0 )
        y_val_pred = np.concatenate( y_val_pred, axis=0 )
        print("Spearman in validation - valence: %f" %spearmanr(y_val_true[:,0], y_val_pred[:,0])[0])
        print("Spearman in validation - energy: %f" %spearmanr(y_val_true[:,1], y_val_pred[:,1])[0])
        print("Spearman in validation - dance: %f" %spearmanr(y_val_true[:,2], y_val_pred[:,2])[0])
        print()
        


# In[63]:


model = torch.load('./multitask_learning')

test_loss, (y_test_gold, y_test_pred) = eval_dataset_regression_multi(val_loader_all, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(y_test_true.shape)

print("Spearman: %f" %spearmanr(y_test_true[:,0], y_test_pred[:,0])[0])
print("Spearman: %f" %spearmanr(y_test_true[:,1], y_test_pred[:,1])[0])
print("Spearman: %f" %spearmanr(y_test_true[:,2], y_test_pred[:,2])[0])

print()


# - For kaggle submissions

# In[64]:


# Define a Pytorch dataset for the test set

class MultitaskDatasetTest(Dataset):
    def __init__(self, path, max_length=-1, read_spec_fn=read_fused_spectrogram, label_type='energy'):
        p = os.path.join(path, 'test')
        self.label_type = label_type
        self.feats = []
        self.files = []
        for f in os.listdir(p):
            self.feats.append(read_spec_fn(os.path.join(p, f)))
            self.files.append(f.split('.')[0])
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length) 

    def __getitem__(self, item):
        # Return a tuple in the form (padded_feats, valence, energy, danceability, length)
        l = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), l, self.files[item]

    def __len__(self):
        return len(self.feats)


# In[65]:


test_specs_multi_all = MultitaskDatasetTest(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         max_length=1293,
         read_spec_fn=read_mel_spectrogram)


test_loader_all = DataLoader(test_specs_multi_all, batch_size=32)


# - Evaluate on the test set

# In[66]:


model = torch.load('./multitask_learning')

y_pred = []  # the predicted labels
names = []

# Obtain the model's device ID
device = next(model.parameters()).device

with torch.no_grad():
    for index, batch in enumerate(test_loader_all, 1):
        # Get the inputs (batch)
        inputs, lengths, files = batch

        # Step 1 - move the batch tensors to the right device
        inputs = inputs.to(device)

        # Step 2 - forward pass: y' = model(x)
        #y_preds = model(inputs, lengths)  # EX9

        # Step 3 - compute loss: L = loss_function(y, y')
        # We compute the loss only for inspection (compare train/test loss)
        # because we do not actually backpropagate in test time
            
        y_preds_val, y_preds_energy, y_preds_dance = model(inputs, lengths)


        
        # Step 5 - collect the predictions, gold labels and batch loss
        y_pred.append(np.hstack((y_preds_val.cpu().numpy(), y_preds_energy.cpu().numpy(), y_preds_dance.cpu().numpy())))
        
        # Step 5 - collect the predictions, gold labels and batch loss
        #y_pred.append(y_preds.cpu().numpy())
        names.append(files)

y_test_pred = np.concatenate( y_pred, axis=0 )
y_test_names = np.concatenate( names, axis=0 )


# - Save the results in the kaggle format.

# In[67]:


with open('./solution.txt', 'w') as f:
    f.write('Id,valence,energy,danceability\n')
    for i, name in enumerate(y_test_names):
        f.write(name + ',' + str(y_test_pred[i,0]) + "," + str(y_test_pred[i,1]) + ',' + str(y_test_pred[i,2]) + '\n')
    

