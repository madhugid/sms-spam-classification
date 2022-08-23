#!/usr/bin/env python
# coding: utf-8

# In[90]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# In[91]:


msg = pd.read_csv('/home/gsunilmadhusudanreddy/Training/NLP/UCI sms spam classifiction/SMSSpamCollection', sep = '\t', names = ['label', 'message'])


# In[92]:


msg.head()


# In[93]:


ps = PorterStemmer()
wordnet = WordNetLemmatizer()


# In[94]:


corpus = []


# In[95]:


for i in range(0, len(msg)):
    message = re.sub('[^a-zA-Z]', ' ', msg['message'][i])
    message = message.lower()
    message = message.split()
    message = [ps.stem(word) for word in message if not word in stopwords.words('english')]
    message = ' '.join(message)
    corpus.append(message)    


# Creating a Bag of words model

# In[96]:


from sklearn.feature_extraction.text import CountVectorizer


# In[97]:


cv = CountVectorizer(max_features = 5000)
X = cv.fit_transform(corpus).toarray()


# In[98]:


y = pd.get_dummies(messages['label'])
y = y.iloc[:,1].values


# In[99]:


from sklearn.model_selection import train_test_split


# In[100]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[101]:


from sklearn.naive_bayes import MultinomialNB


# In[102]:


spam_model = MultinomialNB().fit(X_train, y_train)


# In[103]:


y_pred = spam_model.predict(X_test)


# In[104]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[105]:


conf_m = confusion_matrix(y_test, y_pred)
conf_m


# In[106]:


accuracy = accuracy_score(y_test, y_pred)
round(accuracy,2)


# tfidf model

# In[109]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[110]:


cv_tf = TfidfVectorizer()
X_tf = cv_tf.fit_transform(corpus).toarray()


# In[111]:


y_tf = pd.get_dummies(messages['label'])
y_tf = y_tf.iloc[:,1].values


# In[112]:


X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


# In[113]:


spam_model_tf = MultinomialNB().fit(X_tf_train, y_tf_train)


# In[114]:


y_pred_tf = spam_model.predict(X_tf_test)


# In[115]:


conf_m = confusion_matrix(y_tf_test, y_pred_tf)
conf_m


# In[116]:


accuracy = accuracy_score(y_tf_test, y_pred_tf)
round(accuracy,2)

