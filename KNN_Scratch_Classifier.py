
'''
IMPORTING SCIPY
---------------
'''

from scipy.spatial import distance
def dis(x,y):
    return distance.euclidean(x,y)

'''
BUILDING CLASSIFIER
--------------------
'''

class KNN_Classifier():
    def fit(self,xtrain,ytrain):
        self.xtrain=xtrain
        self.ytrain=ytrain

    def predict(self,xtest):
        pre=[]
        for row in xtest:
            label=self.result(row)
            pre.append(label)
        return pre

    def result(self,row):
        best=dis(row,self.xtrain[0])
        ind=0
        for i in range(1,len(self.xtrain)):
            new_dis=dis(row,self.xtrain[i])
            if new_dis<best:
                best=new_dis
                ind=i
        return self.ytrain[ind]



'''
IMPROTING DEPENDENCIES
----------------------
'''

from sklearn import datasets
from sklearn.metrics import accuracy_score

dataset=datasets.load_iris()


'''
DESCRIBING IRIS DATASET
-----------------------

There are 50 rows for each class of flower.
There is in total of 150 rows of data.
50 rows each for each class.

'''


x=dataset.data
y=dataset.target



'''
SPLITTING DATASET INTO TRAINING ANG TESTING
--------------------------------------------
'''

'''using 50% data for each type of classes for training'''
xtrain=x[0:25]+x[50:75]+x[100:125]
'''using 50% data for each type of classes for testing'''
xtest=x[25:50]+x[75:100]+x[125:150]
ytrain=y[0:25]+y[50:75]+y[100:125]
ytest=y[25:50]+y[75:100]+y[125:150]



'''
USING OUR CLASSIFIER
--------------------
'''

classifier=KNN_Classifier()
classifier.fit(xtrain,ytrain)
prediction=classifier.predict(xtest)


'''
CHECKING ACCURACY OF CLASSIFIER
-------------------------------
'''

print("Accuracy :",int(accuracy_score(ytest,prediction)*100),"%")