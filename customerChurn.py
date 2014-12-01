
# coding: utf-8

# In[69]:

import pandas as pd
import numpy as np
from __future__ import division
from sklearn.preprocessing import StandardScaler


# In[70]:

churn_df = pd.read_csv('path')


# In[71]:

#pulls column names
col_names = churn_df.columns.tolist()


# In[73]:

#selects last twelve columns which are the columns pertinent for ML analysis  
to_show = col_names[-12:]


# In[76]:

# Isolate target data
churn_result = churn_df['Canceled?']
y = np.where(churn_result == 'Canceled',1,0)

y


# In[77]:

#selects unique ID column 
to_drop = ['columns, to , drop']


# In[78]:

#drops previous selected unique ID columns to avoid overfitting 

churn_feat_space = churn_df.drop(to_drop,axis=1)


# In[80]:

#turns all the rows into sepearate arrays. In other words, all the measurements (risk factors, etc) for each customer account 
# are turned into seperate arrays for each customer.
X = churn_feat_space.as_matrix().astype(np.float)
X


# In[81]:

from sklearn.preprocessing import StandardScaler


# In[82]:

#pulls the normalization operation from sklearn library
scaler = StandardScaler()
scaler


# In[83]:

#transforms all the different variables to the same range of 1.0 to -1.0
#or to phrase differently it transforms the individual arrays to a normal distribution centered around 0.
#important step in Machine learning as it is a common requirement for many machine learning estimators
X = scaler.fit_transform(X)

X


# In[84]:

#prints the number of colums by number of rows. Gives dimension of dataframe
print "Feature space holds %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)


# In[85]:

#subsets data to train and test


from sklearn.cross_validation import KFold

def run_cv(X,y,clf_class,**kwargs):
    # Construct a kfolds object
    kf = KFold(len(y),n_folds=5,shuffle=True)
    y_pred = y.copy()
    
    # Iterate through folds
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        y_pred[test_index] = clf.predict(X_test)
    return y_pred



# In[86]:

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN


# In[87]:

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)


# In[88]:

print "Support vector machines:"
print "%.3f" % accuracy(y, run_cv(X,y,SVC))
print "Random forest:"
print "%.3f" % accuracy(y, run_cv(X,y,RF))
print "K-nearest-neighbors:"
print "%.3f" % accuracy(y, run_cv(X,y,KNN))


# In[67]:

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as pl

y = np.array(y)
class_names = np.unique(y)

confusion_matrices = [
    ( "Support Vector Machines", confusion_matrix(y,run_cv(X,y,SVC)) ),
    ( "Random Forest", confusion_matrix(y,run_cv(X,y,RF)) ),
    ( "K-Nearest-Neighbors", confusion_matrix(y,run_cv(X,y,KNN)) ),
]

sv_con_matrix = confusion_matrix(y,run_cv(X,y,SVC))

sv_con_matrix


# In[68]:

conf_arr = confusion_matrix(y,run_cv(X,y,SVC))

norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width = len(conf_arr)
height = len(conf_arr[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
    

cb = fig.colorbar(res, ticks=[ 0, 16000])

plt.show()





# Show confusion matrix
#pl.matshow(cm)
#pl.title('Confusion matrix')
#pl.colorbar()
#pl.show(


# In[ ]:

import matplotlib.pyplot as plt
import numpy as np

x = 14000
y = 200 

# This is the ROC curve
plt.plot(x,y)
plt.show() 


# In[ ]:

def run_prob_cv(X, y, clf_class, **kwargs):
    kf = KFold(len(y), n_folds=5, shuffle=True)
    y_prob = np.zeros((len(y),2))
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
    return y_prob
      


# In[ ]:

import warnings
warnings.filterwarnings('ignore')

# Use 10 estimators so predictions are all multiples of 0.1
pred_prob = run_prob_cv(X, y, RF, n_estimators=10)
pred_churn = pred_prob[:,1]
is_churn = y == 1

# Number of times a predicted probability is assigned to an observation
counts = pd.value_counts(pred_churn)

# calculate true probabilities
true_prob = {}
for prob in counts.index:
    true_prob[prob] = np.mean(is_churn[pred_churn == prob])
    true_prob = pd.Series(true_prob)

# pandas-fu
counts = pd.concat([counts,true_prob], axis=1).reset_index()
counts.columns = ['pred_prob', 'count', 'true_prob']
counts


# In[ ]:

from sklearn.cross_validation import train_test_split

train_index,test_index = train_test_split(churn_df.index)
clf_RF = RF()
clf_RF.fit(X[train_index],y[train_index])


# In[ ]:

train_index,test_index = train_test_split(churn_df.index)
clf_SVC = SVC(probability=True) 
clf_SVC.fit(X[train_index],y[train_index])


# In[ ]:

train_index,test_index = train_test_split(churn_df.index)
clf_KNN = KNN() 
clf_KNN.fit(X[train_index],y[train_index])


# In[ ]:

churn_prob_KNN = clf_KNN.predict_proba(X)
churn_prob_RF = clf_RF.predict_proba(X)
churn_prob_SVC = clf_SVC.predict_proba(X)


# In[ ]:

churn_df['Probability of Churn_SVC']=churn_prob_SVC[:,1]
churn_df['Probability of Churn_KNN']=churn_prob_KNN[:,1]
churn_df['Probability of Churn_RF']=churn_prob_RF[:,1]


# In[1]:

#cds path
import os
os.chdir("path")
churn_df.to_csv("csv_name.csv")


# In[ ]:




# In[ ]:



