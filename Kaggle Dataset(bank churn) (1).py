#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read The Dataset

# In[2]:


df=pd.read_csv("train.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.select_dtypes(include=np.number).columns


# In[5]:


df.select_dtypes(exclude=np.number).columns


# ### check for the null values

# In[6]:


df.isnull().sum()


# In[7]:


### five point summary


# In[8]:


df.describe()


# In[9]:


df["CustomerId"].nunique()


# In[10]:


df["id"].nunique()


# In[11]:


df.duplicated(subset='CustomerId').sum()


# In[12]:


df.drop(columns="id",inplace=True)


# In[13]:


df["Geography"].value_counts()


# In[14]:


df["Gender"].value_counts()


# In[15]:


df["Surname"].value_counts()


# In[16]:


df["CustomerId"].value_counts()


# ### Visulisation

# In[17]:


import plotly.express as px
fig = px.sunburst(
    df,
    path=['Geography','Gender','NumOfProducts','IsActiveMember','Exited'], 
    color='Exited',color_discrete_map={'1':'gold', '0':'darkblue'},
    width=1200, height=1200
)
fig.show()


# In[18]:


plt.figure(figsize=(5, 5))
s=sns.barplot(df,y='NumOfProducts',x='Exited',hue='Gender')
plt.figure(figsize=(5, 5))
s=sns.barplot(df,y='NumOfProducts',x='Exited',hue='Geography', palette='Greens')
plt.figure(figsize=(5, 5))
s=sns.barplot(df,y='Age',x='Exited',hue='Geography',palette='Blues')
plt.figure(figsize=(5, 5))
s=sns.barplot(df,y='Tenure',x='Exited',hue='Geography',palette='Reds')
plt.figure(figsize=(5, 5))
s=sns.scatterplot(df,y='Balance',x='Geography',hue='Exited',palette='crest')


# In[19]:


import warnings
warnings.filterwarnings("ignore")


# In[20]:


corr = df.corr()
# plot the heatmap
plt.figure(figsize=(20, 20))
s=sns.heatmap(corr,annot=True)
corr = df.corr()
# plot the heatmap
plt.figure(figsize=(20, 20))
s=sns.heatmap(corr,annot=True, cmap='plasma')


# In[21]:


#droping columns according to reqirement


# In[22]:


df.drop(columns=["Surname","CustomerId"],inplace=True)


# In[23]:


df.describe()


# In[24]:


df['cred_score']=df["CreditScore"].apply(lambda x: "350-450" if 450>=x>=350 else "451-550"
                       if 550>=x>451 else "551-650"
                       if 650>=x>551 else "651-750"
                       if 750>=x>651 else "751-850")


# In[25]:


df.groupby("cred_score")["Exited"].value_counts()


# In[26]:


df.drop(columns="CreditScore",inplace=True)


# ### train test split

# In[27]:


x=df.drop(columns="Exited")
y=df["Exited"]


# In[28]:


x_num=x.select_dtypes(include=np.number)
x_cat=x.select_dtypes(exclude=np.number)


# In[29]:


dummy_x=pd.get_dummies(x_cat,drop_first=True)


# In[30]:


#scaling


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[32]:


sc=StandardScaler()


# In[33]:


scaled_x=pd.DataFrame(sc.fit_transform(x_num),columns=x_num.columns)


# In[34]:


x=pd.concat([scaled_x,dummy_x],axis=1)


# In[35]:


from sklearn.model_selection import train_test_split


# In[36]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y)


# ### Logistic Regression model

# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


lr=LogisticRegression()


# In[39]:


lr_model=lr.fit(xtrain,ytrain)


# In[40]:


ypred_test=lr_model.predict(xtest)


# In[41]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve,roc_auc_score,precision_score,\
recall_score,f1_score,ConfusionMatrixDisplay,cohen_kappa_score


# In[42]:


print(classification_report(ytest,ypred_test))


# In[43]:


score_card_without_smote = pd.DataFrame(columns=['Model name', 'AUC Score', 'Precision Score', 'Recall Score',
                                       'Accuracy Score', 'Kappa Score', 'f1-score'])

def update_score_card(model, name):
    
    y_pred_train = model.predict(xtrain)
    y_pred_test = model.predict(xtest)
    
    global score_card_without_smote

    score_card_without_smote = score_card_without_smote.append({'Model name' : name,
                                    'AUC Score' : roc_auc_score(ytest, ypred_test),
                                    'Precision Score': precision_score(ytest, ypred_test),
                                    'Recall Score': recall_score(ytest, ypred_test),
                                    'Accuracy Score': accuracy_score(ytest, ypred_test),
                                    'Kappa Score': cohen_kappa_score(ytest, ypred_test),
                                    'f1-score': f1_score(ytest, ypred_test, average='macro')}, 
                                    ignore_index = True)
    
    ConfusionMatrixDisplay.from_predictions(ytest,ypred_test,cmap="Blues")
    plt.show()

    return(score_card_without_smote)


# In[44]:


update_score_card(lr_model, 'Logistic model')


# In[ ]:





# In[ ]:





# ### XGBoost Classifier

# In[45]:


from xgboost.sklearn import XGBClassifier


# In[46]:


xgb_model= XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,gamma=5)
xgb_model.fit(xtrain, ytrain)


# In[47]:


ypred_test=xgb_model.predict(xtest)


# In[48]:


print(classification_report(ytest,ypred_test))


# In[49]:


update_score_card(xgb_model, 'XGboost model')


# ### submission

# In[50]:


test=pd.read_csv("test.csv")


# In[51]:


test.head()


# In[52]:


test.drop(columns=["id","CustomerId","Surname"],inplace=True)


# In[53]:


test['cred_score']=test["CreditScore"].apply(lambda x: "350-450" if 450>=x>=350 else "451-550"
                       if 550>=x>451 else "551-650"
                       if 650>=x>551 else "651-750"
                       if 750>=x>651 else "751-850")


# In[54]:


test.drop(columns="CreditScore",inplace=True)


# In[55]:


scaled_test=pd.DataFrame(sc.fit_transform(test.select_dtypes(include=np.number)),columns=test.select_dtypes(include=np.number).columns)


# In[56]:


dummy_test=pd.get_dummies(test.select_dtypes(exclude=np.number),drop_first=True)


# In[57]:


final_test=pd.concat([scaled_test,dummy_test],axis=1)


# In[58]:


res=xgb_model.predict_proba(final_test)


# In[59]:


test["Exited"]=np.round([i[1] for i in res],1)


# In[60]:


test


# In[ ]:





# In[ ]:





# In[ ]:




