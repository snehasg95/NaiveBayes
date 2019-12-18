import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics

cancer = datasets.load_breast_cancer()
class_name = cancer.target_names
breast_cancer_df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
breast_cancer_df['target'] = cancer.target

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.25, random_state=0)

print(breast_cancer_df.shape)
print(X_train.shape)
print(X_test.shape)

print(y_train.shape)
print(y_test.shape)

clf = GaussianNB()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
acc_score = accuracy_score(y_test, y_predict)

print("The accuracy score is", acc_score)
