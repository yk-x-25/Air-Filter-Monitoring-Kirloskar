# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:23:02 2019

@author: yukes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('kirlos_final_data.csv')
dataset=dataset[:1179]
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,11].values
#finding out no of LABEL 1 and LABEL 0
ct1=0
ct2=0
for i in Y:
    if i==1:
        ct1=ct1+1
    else:
        ct2=ct2+1
#UPsampling LABEL 1
from sklearn.utils import resample
df_majority=dataset[dataset.LABEL==0]
df_minority=dataset[dataset.LABEL==1]

df_min_upsampled=resample(df_minority,replace=True,n_samples=1129,random_state=0)
df_updsampled=pd.concat([df_majority,df_min_upsampled])

df_updsampled.LABEL.value_counts()
y= df_updsampled.LABEL
x = df_updsampled.drop('LABEL', axis=1)



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred) )


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.legend()
plt.show()

'''

#logistic Regression
from sklearn.linear_model import LogisticRegression
clf_1 = LogisticRegression().fit(x, y)
pred_y_2 = clf_1.predict(x)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y,pred_y_2)

from sklearn.metrics import accuracy_score
print(accuracy_score(y, pred_y_2) )
cm_dict={}
cm_dict['Logistic Regression']=accuracy_score(y, pred_y_2)
#KNN
from sklearn.neighbors import KNeighborsClassifier
Classifier=KNeighborsClassifier(n_neighbors =5,metric='minkowski',p=2)
Classifier.fit(x,y)
y_pred_knn=Classifier.predict(x)
print(accuracy_score(y, y_pred_knn) )
cm=confusion_matrix(y,y_pred_knn)

#dictionary for various algorithm
cm_dict={}
cm_dict['Logistic Regression']=accuracy_score(y, pred_y_2)
cm_dict['KNN']=accuracy_score(y, y_pred_knn)



from matplotlib.colors import ListedColormap
X_set, y_set =x,y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
'''
