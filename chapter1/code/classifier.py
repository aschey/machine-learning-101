# https://medium.com/machine-learning-101/chapter-1-supervised-learning-and-naive-bayes-classification-part-2-coding-5966f25f1475
import os
import string
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

train_features = np.load('train_features.npy')
test_features = np.load('test_features.npy')

train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

model = GaussianNB()

print("Training model.")
#train model
model.fit(train_features, train_labels)

predicted_labels = model.predict(test_features)

print("FINISHED classifying. accuracy score : ")
print(accuracy_score(test_labels, predicted_labels))