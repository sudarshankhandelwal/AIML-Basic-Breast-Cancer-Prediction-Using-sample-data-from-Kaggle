#!/usr/bin/env python
# coding: utf-8

# In[76]:


import warnings
warnings.filterwarnings('ignore')


# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[78]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[79]:


df.head()


# In[80]:


df.tail()


# In[81]:


df.columns


# In[82]:


df.info()


# In[83]:


df = df.drop("Unnamed: 32", axis=1)


# In[84]:


df.head()


# In[85]:


df.drop("id", axis = 1, inplace = True)


# In[86]:


df.columns


# In[87]:


type(df.columns)


# In[88]:


l = list(df.columns)
print(l)


# In[89]:


features_mean = l[1:11]
features_se = l[11:21]
features_worst = l[21:]


# In[90]:


print(features_mean)


# In[91]:


print(features_se)


# In[92]:


print(features_worst)


# In[93]:


df.head(2)


# In[94]:


df["diagnosis"].unique()


# In[95]:


df["diagnosis"].value_counts()


# In[96]:


df.shape


# Explore the data

# In[97]:


df.describe()


# In[98]:


len(df.columns)


# In[99]:


corr = df.corr()
corr


# In[100]:


corr.shape


# In[101]:


plt.figure(figsize = (10,10))
sns.heatmap(corr)


# In[102]:


df["diagnosis"] = df["diagnosis"].map({"M":1, "B":0})


# In[103]:


df.head()


# In[104]:


df["diagnosis"].unique()


# In[105]:


X = df.drop("diagnosis", axis = 1)
X.head()


# In[106]:


y = df["diagnosis"]
y.head()


# In[107]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[108]:


df.shape


# In[109]:


X_train.shape


# In[110]:


X_test.shape


# In[111]:


y_train.shape


# In[112]:


y_test.shape


# In[113]:


X_train.head(1)


# In[114]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[115]:


X_train


# Machine learning models

# Logistic Regression

# In[116]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[117]:


y_pred = lr.predict(X_test)


# In[118]:


y_pred


# In[119]:


y_test


# In[120]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[121]:


lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[122]:


results = pd.DataFrame()
results


# In[123]:


tempResults = pd.DataFrame({"Algorithm":["Logistic Regression Method"], "Accuracy":[lr_acc]})
results = pd.concat([results, tempResults])
results = results[["Algorithm", "Accuracy"]]
results


# Decision Tree Classifier

# In[124]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[125]:


y_pred = dtc.predict(X_test)
y_pred


# In[126]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[127]:


dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)


# In[128]:


tempResults = pd.DataFrame({"Algorithm":["Decision Tree Classifier Method"], "Accuracy":[dtc_acc]})
results = pd.concat([results, tempResults])
results = results[["Algorithm", "Accuracy"]]
results


# Random Forest Classifier

# In[129]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[130]:


y_pred = rfc.predict(X_test)
y_pred


# In[131]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[132]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[133]:


tempResults = pd.DataFrame({"Algorithm":["Random Forest Classifier Method"], "Accuracy":[rfc_acc]})
results = pd.concat([results, tempResults])
results = results[["Algorithm", "Accuracy"]]
results


# Support Vector Classifier

# In[134]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train, y_train)


# In[135]:


y_pred = svc.predict(X_test)
y_pred


# In[136]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[137]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[138]:


tempResults = pd.DataFrame({"Algorithm":["Support Vector Classifier Method"], "Accuracy":[svc_acc]})
results = pd.concat([results, tempResults])
results = results[["Algorithm", "Accuracy"]]
results


# In[ ]:





# In[ ]:




