import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn import datasets, svm, metrics,tree
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.decomposition import PCA,KernelPCA,FastICA,NMF,IncrementalPCA,RandomizedPCA
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier
from sklearn import cross_validation
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
import csv

data = pd.read_csv("F:/data/software/GermanData/GermanCredit/credit.txt",delimiter = ',',header=None)

#data=data.sample(0.01)
#data.to_csv("test.csv",index=None)

[row,col]=data.shape
print data.shape
print data.head()
#print data.dtypes
#print data.describe()
#print data.info()



#data = data.drop([5,6], axis=1)
#print data[0].describe()


#print data[col-1].value_counts()


#train=data.sample(frac=0.7)
accuracy=0.0
for i in range(10):
    idx=np.random.permutation(row)
    fen= row*2/3

    train=data.loc[idx[:fen],:col-2]
    trainLabel=data.loc[idx[:fen],col-1]

    le=preprocessing.LabelEncoder()
    trainLabel=le.fit_transform(trainLabel)

    test = data.loc[idx[fen:],:col-2]
    testLabel=data.loc[idx[fen:],col-1]

    testLabel=le.fit_transform(testLabel)


    #print train.info()
    #print test.info()
    for i in train.columns:
        if train[i].dtypes == "object" :
            train[i][train[i].isnull()]=train[i].dropna().mode().values
            test[i][test[i].isnull()]=train[i].dropna().mode().values
        else :
            train[i][train[i].isnull()]=train[i].dropna().median()
            test[i][test[i].isnull()]=train[i].dropna().median()
    #print train.info()
    #print test.info()


    for i in train.columns:
        if train[i].dtypes == "object" :

            train[i] = pd.factorize(train[i])[0]
            test[i] = pd.factorize(test[i])[0]


            '''
            s= str(i)+"class"
            dummy  = pd.get_dummies(train[i],prefix= s)
            train = train.join(dummy)
            train = train.drop([i],axis=1)
            dummy  = pd.get_dummies(test[i],prefix= s)
            test = test.join(dummy)
            test = test.drop([i],axis=1)
            '''

    #print train
    #print train.info()



    pca = PCA(n_components=5)
    #X_pca = pca.fit(train)
    select=SelectKBest(chi2, k=10)
    #X_select = select.fit(train, trainLabel)
    clf = ExtraTreesClassifier(n_estimators=100,max_depth=100)
    X_clf = clf.fit(train, trainLabel)


    #classifier = svm.SVC(gamma=0.001)
    #classifier = tree.DecisionTreeClassifier()
    #classifier = ExtraTreesClassifier(n_estimators=1000, max_depth=100, min_samples_split=1, random_state=0)
    #classifier = RandomForestClassifier(n_estimators=1000, max_depth=100, min_samples_split=1, random_state=0)
    #classifier = LDA()
    #classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=0)
    classifier =LogisticRegression( penalty='l1', tol=0.01)

    X_train=X_clf.transform(train)
    X_test=X_clf.transform(test)
    classifier.fit(X_train, trainLabel)
    predicted = classifier.predict(X_test)
    print("Classification report for classifier %s:\n%s\n"% (classifier, metrics.classification_report(testLabel, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testLabel, predicted))
    accuracy += metrics.accuracy_score(testLabel,predicted)
    print metrics.accuracy_score(testLabel,predicted)

accuracy /=10
print("final average accucacy is %lf" % accuracy)

