## __Lab 1:__ Optical digit recognition

- The scope of the 1st Lab is the implementation of an optical digit recognition system. The dataset comes from US Postal Service and contains digits from 0 to 9 separated in train and test set.
- Technologies used: NumPy, scikit-learn, PyTorch.

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
  
- The main part of the 1st Lab that contains all the results is placed in [Lab1](./Lab1). A jupyter notebook, along with an html and a python export are available. 
