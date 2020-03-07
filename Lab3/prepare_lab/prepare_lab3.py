#!/usr/bin/env python
# coding: utf-8

# # Classify Genres and Emotions in Songs Using Deep Learning

# ## Description:
# 
# The goal of this lab is to recognize the genre and extract the emotions from spectrograms of music songs. We are given 2 datasets:
# - Free Music Archive (FMA) genre that contains 3834 samples from 20 music genres.
# - Multitask music dataset that contains 1497 samples with labels about the emotions such as valence, energy and danceability.
# 
# All samples came from spectrograms, that have been extracted from clips of 30 seconds from different songs.
# 
# We will analyze the spectrograms using deep learning architectures such as Recurrent Neural Networks and Convolutional Neural Networks. The exercise is separated in 5 parts:
# 
# 1. Data analysis and familiarize with spectrograms.
# 2. Implement classifiers about the music genre using the FMA dataset.
# 3. Implement regression models for predicting valence, energy and danceability.
# 4. Use of modern training techniques, such as transfer and multitask learning, to improve the previous results.
# 5. Submit results in the Kaggle competition of the exercise.

# ## Implementation
# 
# In the prepare lab, we will classify music genres using the spectrograms.

# In[1]:


# Import necessary libraries
import numpy as np
import copy
import re
import os
import pandas as pd
import random
import librosa.display
import matplotlib.pyplot as plt
# sklearn
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
# Pytorch
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import SubsetRandomSampler, DataLoader


# ### Step 0: Familiarize with kaggle kernels
# Open a private kernel in kaggle and load the data. Run the command **os.listdir("../input/patreco3-multitask-affective-music/data/")** to check the subfolders of the dataset. Try enabling and disabling the GPU and commit your changes.

# In[2]:


os.listdir("../input/patreco3-multitask-affective-music/data/")


# ## Step 1: Familiarize with spectrograms in mel scale

# 1. Choose two files randomly

# In[3]:


fixed = True

if not fixed:
    # Open train_labels.txt file and choose two random lines (with different labels).
    with open('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt', 'r') as f:
        lines = f.readlines()
        # The first element is the headers.
        train_size = len(lines) - 1
        idx_1 = random.randint(1,train_size)
        filename_1, label_1 = lines[idx_1].split()

        label_2 = label_1
        while (label_2 == label_1):
            idx_2 = random.randint(1,train_size)
            filename_2, label_2 = lines[idx_2].split()
else:
    filename_1 = '63227.fused.full.npy.gz'
    label_1 = 'Blues'
    filename_2 = '28466.fused.full.npy.gz'
    label_2 = 'Jazz'
    
    
print('1st file: %s %s' %(filename_1, label_1))
print('2nd file: %s %s' %(filename_2, label_2))


# - Load the spectograms and keep the mel-spectograms.

# In[4]:


spec_1 = np.load('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/' + filename_1.strip(".gz"))
spec_2 = np.load('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/' + filename_2.strip(".gz"))

print('Shape of 1st spectrogram:')
print(spec_1.shape)
print('Shape of 2nd spectrogram:') 
print(spec_2.shape)

mel_1 = spec_1[:128]
mel_2 = spec_2[:128]


# - Plot the spectrograms.

# In[5]:


plt.rcParams['figure.figsize'] = [30, 10]

plt.subplot(1, 2, 1)
plt.title(label_1, fontsize=30)
librosa.display.specshow(mel_1)
plt.subplot(1, 2, 2)
plt.title(label_2, fontsize=30)
librosa.display.specshow(mel_2)


# In the above spectrograms, the horizontal axis represents time, and the vertical axis represents frequency. A third dimension indicating the amplitude of a particular frequency at a particular time is represented by the intensity of color of each point in the image. If we compare the above spectrograms, we can see that the Blues music has lower frequencies the entire time, while the Jazz music has many changes in the frequency over time. More generally, songs from the same genre will have similar frequency changes over time.

# ## Step 2: beat-synced spectrograms

# - Print the shape of the mel-spectrograms.

# In[6]:


print('Shape of 1st mel-spectrogram:')
print(mel_1.shape)
print('Shape of 2nd mel-spectrogram:') 
print(mel_2.shape)


# We can see that the timesteps of the mel-spectrograms are 1291 and 1291 respectively. If we trained our LSTM in these data, our model will not be efficient enough, since the length of each sample will be very large. In order to resolve this problem, we should synchronize the spectrogram with the beat of the music. This is done by taking the median between the points of the beat. The final files are placed in '/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/'.

# In[7]:


spec_1_beat = np.load('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/' + filename_1.strip(".gz"))
spec_2_beat = np.load('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/' + filename_2.strip(".gz"))

mel_1_beat = spec_1_beat[:128]
mel_2_beat = spec_2_beat[:128]

print('Shape of 1st mel-spectrogram after beat-sync:')
print(mel_1_beat.shape)
print('Shape of 2nd mel-spectrogram after beat-sync:') 
print(mel_2_beat.shape)


# In[8]:


plt.subplot(1, 2, 1)
plt.title(label_1, fontsize=30)
librosa.display.specshow(mel_1_beat)
plt.subplot(1, 2, 2)
plt.title(label_2, fontsize=30)
librosa.display.specshow(mel_2_beat)


# We observe that the changes in the frequency are almost the same, despite the fact that the timesteps are less.

# ## Step 3: Familiarize with chromagrams

# In[9]:


chroma_1 = spec_1[128:]
chroma_2 = spec_2[128:]

print('Shape of 1st chromagram:')
print(chroma_1.shape)
print('Shape of 2nd chromagram:') 
print(chroma_2.shape)

chroma_1_beat = spec_1_beat[128:]
chroma_2_beat = spec_2_beat[128:]

print('Shape of 1st chromagram after beat-sync:')
print(chroma_1_beat.shape)
print('Shape of 2nd chromagram after beat-sync:') 
print(chroma_2_beat.shape)


# In[10]:


plt.subplot(1, 2, 1)
plt.title(label_1, fontsize=30)
librosa.display.specshow(chroma_1)
plt.subplot(1, 2, 2)
plt.title(label_2, fontsize=30)
librosa.display.specshow(chroma_2)


# In[11]:


plt.subplot(1, 2, 1)
plt.title(label_1, fontsize=30)
librosa.display.specshow(chroma_1_beat)
plt.subplot(1, 2, 2)
plt.title(label_2, fontsize=30)
librosa.display.specshow(chroma_2_beat)


# ## Step 4: Load and Analyze data

# - Combine similar classes and remove underrepresented classes. Similar classes are combined because our classifier will find it difficult to distinguish them. On the other hand, classes that are underrepresented will not be correctly recognized because their available training data will not be enough.

# In[12]:


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


# - Split dataset in train and validation set.

# In[13]:


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


# - Define some useful functions for loading spectrograms and chromagrams

# In[14]:


def read_fused_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)
    return spectrogram.T


def read_mel_spectrogram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[:128]
    return spectrogram.T

    
def read_chromagram(spectrogram_file):
    spectrogram = np.load(spectrogram_file)[128:]
    return spectrogram.T


# - Define an encoder for the labels.

# In[15]:


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


# - Define a PaddingTransformer in order to convert all input sequences to the same length.

# In[16]:


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


# - Define Pytorch dataset

# In[17]:


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


# - Load mel spectrograms

# In[18]:


mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_mel, val_loader_mel = torch_train_val_split(mel_specs, 32 ,32, val_size=.33)
test_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_mel = DataLoader(test_mel, batch_size=32)


# - Load beat synced mel spectrograms

# In[19]:


beat_mel_specs = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
train_loader_beat_mel, val_loader_beat_mel = torch_train_val_split(beat_mel_specs, 32 ,32, val_size=.33)
test_beat_mel = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)
test_loader_beat_mel = DataLoader(test_beat_mel, batch_size=32)


# - Load beat synced chromagrams

# In[20]:


beat_chroma = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_chromagram)
train_loader_beat_chroma, val_loader_beat_chroma = torch_train_val_split(beat_chroma, 32 ,32, val_size=.33)
test_beat_chroma = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_chromagram)
test_loader_beat_chroma = DataLoader(test_beat_chroma, batch_size=32)


# - Load fused speectrogram + chromagram for the full (non-beat-synced) data

# In[21]:


specs_fused = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=True,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_fused_spectrogram)
train_loader, val_loader = torch_train_val_split(specs_fused, 32 ,32, val_size=.33)
test = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',
         train=False,
         class_mapping=class_mapping,
         max_length=-1,
         read_spec_fn=read_fused_spectrogram)
test_loader = DataLoader(test, batch_size=32)


# - Display 2 histograms, one before class_mapping and one after.

# In[22]:


# Load the beat sync mel-spectrograms without class mapping.
beat_mel_specs_nomap = SpectrogramDataset(
         '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',
         train=True,
         max_length=-1,
         read_spec_fn=read_mel_spectrogram)


# In[23]:


# Keep all the train labels before the mapping.

labels_before = []
for i in range(len(beat_mel_specs_nomap)):
    _, label, _ = beat_mel_specs_nomap[i]
    labels_before.append(label)


# In[24]:


# Keep all the train labels after the mapping.

labels_after = []
for i in range(len(beat_mel_specs)):
    _, label, _ = beat_mel_specs[i]
    labels_after.append(label)


# In[25]:


# Plot the histograms side by side.

plt.rcParams['figure.figsize'] = [25, 10]
plt.subplot(1, 2, 1)
plt.title('Before class mapping', color='w', fontsize=30)
plt.hist(labels_before, bins=20)
plt.subplot(1, 2, 2)
plt.hist(labels_after, bins=10)
plt.title('After class mapping', color='w', fontsize=30)


# ## Step 5: Music Genre Classification using LSTM

# - Define LSTM

# In[26]:


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.rnn_size = rnn_size
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # --------------- Insert your code here ---------------- #
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
        
        # --------------- Insert your code here ---------------- #
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
        return last_outputs

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


# - Define a function that trains the model for an epoch.

# In[27]:


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


# - Define a function that evaluates the model in an epoch.

# In[28]:


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


# ### Test the model by training only in one batch to make it to overfit.

# For convenience, we save the models and each time we want to analyze them we load them back.

# In[29]:


# Define useful parameters that are the same for all the models.
num_mel = 128
num_chroma = 12
n_classes = 10


# In[30]:


train_batch = next(iter(train_loader_beat_mel))

RNN_SIZE = 32
EPOCHS = 5000

model = BasicLSTM(num_mel, RNN_SIZE, n_classes, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # IMPORTANT: switch to train mode
    # Εnable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0

    # Οbtain the model's device ID
    device = next(model.parameters()).device

    # Get the inputs (batch)
    inputs, labels, lengths = train_batch
    

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
    running_loss = loss.data.item()
    
    if epoch%100 == 0:
        print("Epoch %d with loss: %f" %(epoch, running_loss))

torch.save(model, './overtrained_model')


# ### Train the model in the mel spectrograms

# In[31]:


RNN_SIZE = 32
EPOCHS = 500

model = BasicLSTM(num_mel, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_mel, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_mel, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_mel, model, loss_function)
    
    if epoch%(100) == 0: 
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
        
torch.save(model, './mel_32_500')


# In[32]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_mel, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# ### Train the model in the beat synced mel spectrograms

# In[33]:


RNN_SIZE = 32
EPOCHS = 500

model = BasicLSTM(num_mel, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_beat_mel, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_beat_mel, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_beat_mel, model, loss_function)
    
    if epoch%(100) == 0: 
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
        
torch.save(model, './beat_mel_32_500')


# In[34]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_beat_mel, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# As we can see, all the metrics are higher due to the beat synchronization.

# In[35]:


RNN_SIZE = 64
EPOCHS = 1000

model = BasicLSTM(num_mel, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001,  weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_beat_mel, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_beat_mel, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_beat_mel, model, loss_function)
    
    if epoch%(100) == 0: 
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

torch.save(model, './beat_mel_32_1000')


# In[36]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_beat_mel, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# ### Train the model in the beat synced chromagrams

# In[37]:


RNN_SIZE = 16
EPOCHS = 300

model = BasicLSTM(num_chroma, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_beat_chroma, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_beat_chroma, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_beat_chroma, model, loss_function)
    
    if epoch%(50) == 0: 
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

torch.save(model, './beat_chroma_16_300')


# In[38]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_beat_chroma, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# In[39]:


RNN_SIZE = 32
EPOCHS = 500

model = BasicLSTM(num_chroma, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader_beat_chroma, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader_beat_chroma, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader_beat_chroma, model, loss_function)
    
    if epoch%(50) == 0: 
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

torch.save(model, './beat_chroma_32_500_l2reg')


# In[40]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader_beat_chroma, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# Here, our model overfits because we trained it for too many epochs.

# ### Train the model in the beat fused spectrograms

# In[41]:


RNN_SIZE = 32
EPOCHS = 500

model = BasicLSTM(num_chroma+num_mel, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, loss_function)
    
    if epoch%(100) == 0: 
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
        
torch.save(model, './beat_fused_32_500')


# In[42]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# In[43]:


RNN_SIZE = 64
EPOCHS = 1000

model = BasicLSTM(num_chroma+num_mel, RNN_SIZE, 10, 1, bidirectional=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.double()
model.to(DEVICE)
loss_function = nn.CrossEntropyLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5 )

for epoch in range(EPOCHS):
    # Train the model for one epoch
    train_dataset(epoch, train_loader, model, loss_function, optimizer)
    
    # Evaluate the performance of the model, on both data sets
    train_loss, (y_train_gold, y_train_pred) = eval_dataset(train_loader, model, loss_function)

    val_loss, (y_val_gold, y_val_pred) = eval_dataset(val_loader, model, loss_function)
    
    if epoch%(100) == 0: 
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
        
torch.save(model, './beat_fused_64_1000')


# In[44]:


test_loss, (y_test_gold, y_test_pred) = eval_dataset(test_loader, model, loss_function)

y_test_true = np.concatenate( y_test_gold, axis=0 )
y_test_pred = np.concatenate( y_test_pred, axis=0 )

print(classification_report(y_test_true, y_test_pred))


# ### Step 6: Model evaluation

# First some definitions:
#     
# - Accuracy: Percentage of total items classified correctly. It is a good measure when the target variable classes in the data are nearly balanced.
# - Precision: Number of items correctly identified as positive out of total items identified as positive. It is about being precise,
# - Recall: Number of items correctly identified as positive out of total true positives.
# - f1-score: It is the harmonizc mean between precision and recal
# - macro averaged metrics: It is the arithmetic mean of the per-class metrics
# - micro averaged metrics: It is the weigthed mean of the per-class metrics

# 1. When the difference between the accuracy and the f1-score is big, it means that the dataset is imbalanced.

# 2. When the difference between the macro and micro f1 score is big, it means that the dataset is imbalanced, because the micro f1 weighs the f1 scores according to the size of each class.

# 3. When it is not so much about capturing cases correctly but more about capturing all cases that are in a certain class, recall is a better metric than precision. A classical example is the cancer prediction problem. When we treat false positives and false negatives the same, we can use the accuracy as our classification metric.
