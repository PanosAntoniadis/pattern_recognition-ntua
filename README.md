## __Lab 1:__ Optical digit recognition

- The scope of the 1st Lab is the implementation of an optical digit recognition system. The dataset comes from US Postal Service and contains digits from 0 to 9 separated in train and test set.

- Some results:

  | Classifier | Accuracy |
  | --------- | -------- |
  | Sklearn Gaussian Naive Bayes | 71.9% |
  | NumPy Gaussian Naive Bayes (smooth=1e-3) | 76.1% |
  | NumPy Gaussian Naive Bayes (smooth=1e-3, var=1) | 81.3% |
  | NumPy Euclidean | 81.41% |
  | SVM - poly kernel | 94.7% |
  | SVM - rbf kernel | 92.6% |
  | Nearest Neighbor | 94.5% |
  | Voting (SVM poly + NN + SVM linear)| __95.01%__ |
  | Bagging using SVM poly| __94.71%__ |
  | Neural Network - 2 hidden (128, 64) - tanh| 92.32% |
  
- The main part of the 1st Lab that contains all the results is placed in [Lab1](./Lab1). A [jupyter notebook](./Lab1/main_lab1.ipynb), along with an [html](./Lab1/main_lab1.html) and a [python](./Lab1/main_lab1.py) export are available. 

## __Lab 2:__ Speech Recognition using HMMs and RNNs

-  The scope of the 2nd Lab is the implementation of a speech recognition system, that recognizes isolated digits. The dataset is the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).

- Models used
  - GMM-HMMs with different number of states ano Gaussians
  - LSTM
  - LSTM with Dropout and L2-Regularization
  - LSTM with Early Stoping
  - BiLSTM
  - BiLSTM with pack_padded_sequence


- Best results:

  | Classifier | Accuracy |
  | --------- | -------- |
  | Best GMM-HMM model | 96% |
  | Best LSTM model | 94% |


## __Lab 3:__ Genre Classification and Sentiment prediction from music

- The scope of the 3rd Lab is to recognize the genre and extract the emotions from spectrograms of music songs. Two datasets are used: Free Music Archive (FMA) and Multitask music dataset ([kaggle link](https://www.kaggle.com/geoparslp/patreco3-multitask-affective-music)).

### Genre Classification

- Models used:
  - LSTM in spectrograms
  - LSTM after applying beat syncing in spectrograms
  - LSTM in chromagrams
  - LSTM after applying beat syncing in chromagrams
  - LSTM after applying beat syncing in fused spectrograms
  - CNN with 4 layers (Conv - Batch - ReLU - Pool)

### Sentiment prediction

- Models used:
  - LSTM 
  - CNN
  - Transfer Learning using the CNN of genre classification
  - CNN with Multitask Learning

## Technologies used: 
- NumPy
- Scikit-learn
- PyTorch
- Matplotlib
- Librosa
