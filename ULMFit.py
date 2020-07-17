#!/usr/bin/env python
# coding: utf-8

# ## ULMfit sentiment model

# **Importing necessary libraries**

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import fastai
import wget
from fastai import *
from fastai.text import * 
import pandas as pd
import numpy as np
from functools import partial
import io
from fastai.text import * 
from fastai.callbacks import *
import os


# In[3]:


data = pd.read_csv('B:/Projects/ULMfit model/Tweets.csv')


# In[4]:


data.shape


# In[5]:


data.head()


# ## Part 1 - Data Analysis

# In[6]:


total_null_values = data.isnull().sum().sum()
total_values = data.shape[0]*data.shape[1]
print('Total null values = {}'.format(total_null_values))
print('Total values = {}'.format(total_values))
print("% null values = {}".format(total_null_values/total_values*100))


# In[7]:


data.describe()


# In[8]:


# Gist of null values column wise

data.isnull().sum()


# ## Part 2 - Data Cleaning

# Dropping `unnecessary` columns

# In[9]:


data.drop('airline_sentiment_gold', axis=1, inplace=True)
data.drop('negativereason_gold', axis=1, inplace=True)
data.drop('tweet_coord', axis=1, inplace=True)
data.drop('tweet_location', axis=1, inplace=True)
data.drop('user_timezone', axis=1, inplace=True)


# In[10]:


data.isnull().sum()


# ###### Null values left after dropping some columns

# In[11]:


data.isnull().sum().sum()


# `negativereason_confidence`, `negativereason` **is an important column so we need to fill it with some relevant values**

# In[12]:


data['negativereason_confidence'] = data['negativereason_confidence'].fillna(data['negativereason_confidence'].mean())


# In[13]:


data.isnull().sum().sum()


# In[14]:


data['negativereason'] = data['negativereason'].fillna('Not Negative')


# In[15]:


data.isnull().sum()


# In[18]:


data.isnull().sum().sum()


# **Now we have successfully cleaned our data as there are no null values left**

# In[52]:


data.head()


# ## Part 3 - WordCloud

# In[19]:


negative=data[data['airline_sentiment']=='negative']
words = ' '.join(negative['text'])
cleaned_word = " ".join([word for word in words.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])


# In[20]:


wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500
                     ).generate(cleaned_word)


# In[21]:


cleaned_word


# In[22]:


plt.figure(1,figsize=(8, 14))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# ## Part 4 - Data Visualization

# In[23]:


plt.style.use('seaborn')
sns.countplot(data=data,x='airline_sentiment')


# In[24]:


plt.bar(data['airline'], data['retweet_count'], color='green')
plt.xlabel('Airline')
plt.ylabel('Retweet-count')


# In[25]:


data.hist()


# ## Part 5 - Data Modelling

# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:



# split data into training and validation set
df_trn, df_val = train_test_split(data, test_size = 0.4, random_state = 12)


# In[28]:


df_trn.shape, df_val.shape


# In[29]:


# Language model data
data_lm = TextLMDataBunch.from_df(train_df = df_trn, valid_df = df_val, path = "")

# Classifier model data
data_clas = TextClasDataBunch.from_df(path = "", train_df = df_trn, valid_df = df_val, vocab=data_lm.train_ds.vocab, bs=32)


# In[30]:


#learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn = language_model_learner(data_lm,  arch = AWD_LSTM, pretrained = True, drop_mult=0.3)


# In[31]:


learn.lr_find()


# In[32]:


learn.recorder.plot(suggestion=True)


# In[33]:


learn.fit_one_cycle(2, 5.75E-02,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))


# In[34]:


learn.save('fit_head')


# In[35]:


learn.unfreeze()


# In[36]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[37]:


learn.fit_one_cycle(3,3.98E-04,callbacks=[SaveModelCallback(learn, name="best_lm")], moms=(0.8,0.7))


# In[38]:


learn.load('best_lm')


# In[39]:


learn.save_encoder('AIBoot_enc')


# In[40]:


learn1 = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.3)


# In[41]:


learn1.load_encoder('AIBoot_enc')


# In[42]:


learn.fit_one_cycle(3, 1e-2)


# In[43]:


learn.save_encoder('ft_enc')


# In[46]:


learn = text_classifier_learner(data_clas, arch=AWD_LSTM, drop_mult=0.7)
learn.load_encoder('ft_enc')
learn.fit_one_cycle(2, 1e-2)


# In[47]:


# get predictions
preds, targets = learn.get_preds()

predictions = np.argmax(preds, axis = 1)
pd.crosstab(predictions, targets)


# ### Thank you for providing me this opportunitity.
# #### Though it was my first time working with `ULMfit` model, I enjoyed alot learning about it and I tried my best.
# Once again, Thanks a alot!

# In[ ]:




