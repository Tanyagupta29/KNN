import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 


data = pd.read_csv(r'C:\KNN algo\train.csv')




X = data.drop(['label'],axis = 1)
y = data['label']

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y = train_test_split(X,y,test_size = 0.3, random_state = 10)

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn import metrics


def elbow(k):

    
    error_test = []

   
    for i in k:
       
        clf = knn(n_neighbors=i)
        clf.fit(train_x,train_y)
         
        tmp = clf.predict(test_x)
        tmp = metrics.accuracy_score(tmp,test_y)
        error = 1-tmp
        error_test.append(error)
    return error_test


k = range(1,10)


test = elbow(k)



plt.plot(k, test)
plt.xlabel('K Neighbors')
plt.ylabel('Test error')
plt.title('Elbow curve for test')
plt.show()




m={}
for i in range(1,10):
    m[i]=np.interp(i,k,test)

val=1
for j in range(1,10):
    if(val>m[j]):
        val=m[j]
        num=j


clf = knn(n_neighbors=num) 
clf.fit(train_x,train_y)

pred = clf.predict(test_x)

from sklearn.metrics import classification_report
print(classification_report(test_y, pred)) 