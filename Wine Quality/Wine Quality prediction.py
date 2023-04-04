#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt # we only need pyplot
sb.set() # set the default Seaborn style for graphics
import plotly.express as px


# In[20]:


winedata = pd.read_csv('winequality-red.csv')
winedata.head()


# In[21]:


winedata.info()


# In[26]:



fig = px.histogram(winedata,x='quality')
fig.show()


# In[23]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="fixed acidity", data=winedata)


# In[22]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="volatile acidity", data=winedata)


# In[205]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="citric acid", data=winedata)


# In[206]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="residual sugar", data=winedata)


# In[207]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="chlorides", data=winedata)


# In[208]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="free sulfur dioxide", data=winedata)


# In[209]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="total sulfur dioxide", data=winedata)


# In[210]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="sulphates", data=winedata)


# In[211]:


fig = plt.figure(figsize = (6,3))
sb.barplot(x="quality", y="alcohol", data=winedata)


# In[212]:


f = plt.figure(figsize=(25,25))
sb.heatmap(winedata.corr(), vmin=-1, vmax=1, linewidths=1, annot = True, fmt = ".2f", annot_kws = {"size": 18}, cmap = "RdBu")


# In[213]:


new = []
for row in winedata['quality']:
    if (row <= 5):
        val = 'Bad'
    else:
        val = 'Good'
    new.append(val)

winedata['newquality'] = new

from collections import Counter
print(sorted(Counter(winedata['newquality']).items()))


# In[214]:


sb.countplot(winedata['newquality'])


# In[215]:


X = winedata.drop(['quality', 'newquality'], axis = 1)
y = winedata['newquality']


# In[216]:


from sklearn.model_selection import train_test_split
x_train1,x_test,y_train1,y_test = train_test_split(X, y, test_size = 0.20, random_state = 1234)


# In[217]:


from imblearn.over_sampling import SMOTE

oversample = SMOTE()
x_train, y_train = oversample.fit_resample(x_train1, y_train1)


# In[218]:


sb.countplot(y_train)


# In[225]:


print(sorted(Counter(y_train).items()))


# In[219]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[226]:


#decision tree classifier
from sklearn.tree import plot_tree
dt = DecisionTreeClassifier()
model1 = dt.fit(x_train, y_train)


print("train accuracy: ", model1.score(x_train, y_train), "\n", "test accuracy: ", model1.score(x_test, y_test))

dtpred = dt.predict(x_test)
print("\n")
print("classification report for decision tree classifier")
print(classification_report(dtpred, y_test))
print("\n")
print("confusion matrix for decision tree classifier")
displr = plot_confusion_matrix(dt, x_test, y_test, cmap = plt.cm.OrRd, values_format = 'd')


# In[227]:


#random forest classifier
rf = RandomForestClassifier()
model2 = rf.fit(x_train, y_train)
print("train accuracy:",model1.score(x_train, y_train),"\n","test accuracy:",model2.score(x_test,y_test))

rfpred = rf.predict(x_test)
print("\n")
print("classification report for random forest classifier")
print(classification_report(rfpred,y_test))
print("\n")
print("confusion matrix for random forest classifier")
displr = plot_confusion_matrix(rf, x_test, y_test ,cmap=plt.cm.OrRd , values_format='d')


# In[228]:


#logistic regression
lr = LogisticRegression(max_iter=20000,penalty='l2')
model3=lr.fit(x_train, y_train)
print("train accuracy:",model1.score(x_train, y_train),"\n","test accuracy:",model1.score(x_test,y_test))

lrpred = lr.predict(x_test)
print("\n")
print("classification report for logistic regression")
print(classification_report(lrpred,y_test))
print("\n")
print("confusion matrix for logistic regression")
displr = plot_confusion_matrix(lr, x_test, y_test,cmap=plt.cm.OrRd , values_format='d')


# In[223]:


# adaboost classifier 
ada=AdaBoostClassifier()
model4=ada.fit(x_train, y_train)
print("train accuracy:",model4.score(x_train, y_train),"\n","test accuracy:",model4.score(x_test,y_test))

adapred = ada.predict(x_test)
print("\n")
print("classification report for adaboost classifier")
print(classification_report(adapred,y_test))
print("\n")
print("confusion matrix for adaboost classifier")
displr = plot_confusion_matrix(ada, x_test, y_test ,cmap=plt.cm.OrRd , values_format='d')


# In[224]:


#support vector classifier
svc=SVC()
model5=svc.fit(x_train, y_train)
print("train accuracy:",model5.score(x_train, y_train),"\n","test accuracy:",model5.score(x_test,y_test))

svcpred = svc.predict(x_test)
print("\n")
print("classification report for support vector classifier")
print(classification_report(svcpred,y_test))
print("\n")
print("confusion matrix for support vector classifier")
displr = plot_confusion_matrix(svc, x_test, y_test ,cmap=plt.cm.OrRd , values_format='d')


# In[ ]:




