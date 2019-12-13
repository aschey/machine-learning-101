from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np

train_features = np.load('train_features.npy')
test_features = np.load('test_features.npy')

train_labels = np.load('train_labels.npy')
test_labels = np.load('test_labels.npy')

model = svm.SVC()
model.fit(train_features, train_labels)

predicted_labels = model.predict(test_features)
print(accuracy_score(test_labels, predicted_labels))