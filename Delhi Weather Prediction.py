import numpy as np
import pandas as pd

data=pd.read_csv("simpler_testset.csv",header=0)

data=data.replace({"_dewptm":np.NaN,"_tempm":np.NaN,"_pressure":-9999},data.interpolate())

data=data.replace({"_wspdm":np.NaN},0)

#print(data.shape)

data=data.dropna()

#print(data.shape)

X=data.iloc[:,:-1]
Y=data.iloc[:,-1]


#print(len(Y))

from sklearn import  svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

classifier=svm.SVC(gamma=0.000001)


X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.9888)

classifier.fit(X_train,y_train)

prediction=classifier.predict(X_test)

print("Accuracy :",float(accuracy_score(y_test,prediction)*100),"%")
