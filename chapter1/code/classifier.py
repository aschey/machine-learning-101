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

# stopwords = set(stopwords.words('english') + list(string.punctuation))
# include_subject = True

# def read_data(folder):
#     data = []
#     labels = []
#     files = (os.path.join(folder, f) for f in os.listdir(folder))
#     p = PorterStemmer()
#     for fil in files:
#         with open(fil, 'r') as f:
#             if include_subject:
#                 line = ' '.join(f.readlines())
#             else:
#                 line = f.readlines()[2]
#             data.append([p.stem(w) for w in word_tokenize(line) if w not in stopwords and w.isalpha() and len(w) > 1])

#             if "spmsg" in fil:
#                 labels.append(1)
#             else:
#                 labels.append(0)
#     return data, labels


# def clean_data(file_data, top):
#     words = [w for data in file_data for w in data]
#     if top is not None:
#         return [k[0] for k in Counter(words).most_common(top)]
#     return words

# def extract_features(file_data, words):
    
#     features = np.zeros((len(file_data), len(words)))
#     for i, data in enumerate(file_data):
#         for j, word in enumerate(data):
#             try:
#                 features[i,words.index(word)] = data.count(word)
#             except:
#                 continue
#     return features

# train_folder = 'chapter1/train-mails'
# test_folder = 'chapter1/test-mails'
# train_data, train_labels = read_data(train_folder)
# test_data, test_labels = read_data(test_folder)

# count = clean_data(train_data, 2000)
# train_features = extract_features(train_data, count)
# test_features = extract_features(test_data, count)
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