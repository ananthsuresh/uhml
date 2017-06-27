import numpy as np
import urllib
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score

#loading breast cancer data from UCI ML Repo
url = "https://goo.gl/AP7kzV"
raw_data = urllib.request.urlopen(url)
dataset = np.genfromtxt(raw_data, delimiter=",")

#features are first 9, and classification is the 10th
X = dataset[:,0:10]
y = dataset[:,10]

#splits data into 5 chunks for N fold cross validation
k_fold = KFold(n_splits=5)
#below chunk is to test which are testing and training
#for train_indices, test_indices in k_fold.split(X):
    #print('Train: %s | test: %s' % (train_indices, test_indices))

svc = svm.SVC(C=1, kernel='linear')

#to change all nan values to 0
X[(np.isnan(X))] = 0

#loop to do 5 fold cross validation, stores scores in array
# scores = [svc.fit(X[train], y[train]).score(X[test], y[test])
#     for train, test in k_fold.split(X)]
# print(scores)
i = 1

for train, test in k_fold.split(X):
    fitted = svc.fit(X[train], y[train])
    accuracy = fitted.score(X[test], y[test])
    predicted = svc.predict(X[test])
    report = classification_report(y[test], predicted)
    confusion = confusion_matrix(y[test], predicted)
    f1 = f1_score(y[test], predicted)
    print("Scores for Fold %d \n", i)
    print(report)
    print("\n")
    print("Accuracy Score: %f \n",accuracy)
    print("Confusion Matrix: \n")
    print(confusion)
    print("\n")
    print("F1 score: %f \n", f1)
    i+=1
