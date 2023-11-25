#!/usr/bin/env python
# coding: utf-8

# In[61]:


import pandas as pd


# In[62]:


import os
print(os.getcwd())


# In[63]:


df = pd.read_csv("C:\spotify_millsongdata.csv")


# In[64]:


df.head(5)


# In[65]:


df.tail(5)


# In[66]:


df.shape


# In[67]:


df.isnull().sum()


# In[68]:


df =df.sample(5000).drop('link', axis=1).reset_index(drop=True)


# In[69]:


df.head(10)


# In[70]:


# df = df.sample(5000)


# In[71]:


df['text'][0]


# In[72]:


df=df.sample(5000)


# In[73]:


df.shape


# Text Cleaning/ Text Preprocessing

# In[74]:


df['text'] = df['text'].str.lower().replace(r'^\w\s', ' ').replace(r'\n', ' ', regex = True)


# In[75]:


import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)


# In[76]:


df['text'] = df['text'].apply(lambda x: tokenization(x))


# In[77]:


import nltk
nltk.download('punkt')


# In[78]:


import nltk
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)


# In[79]:


df['text'] = df['text'].apply(lambda x: tokenization(x))


# In[80]:


tfidvector = TfidfVectorizer(analyzer='word',stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)


# In[81]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[82]:


tfidvector = TfidfVectorizer(analyzer='word',stop_words='english')
matrix = tfidvector.fit_transform(df['text'])
similarity = cosine_similarity(matrix)


# In[83]:


similarity[0]


# In[90]:


df[df['song'] == 'Heartbreak Hotel']


# In[85]:


print(len(df))
print(df.shape[0])



# In[91]:


print(df['song'].isin(['Heartbreak Hotel']).any())


# In[93]:


def recommendation(song_df):
    if len(df) == 0 or not df['song'].isin([song_df]).any():
        print("Song not found in DataFrame.")
        return

    idx = df[df['song'] == song_df].index[0]
    # Rest of your code...


# In[94]:


def recommendation(song_df):
    idx = df[df['song'] == song_df].index[0]
    distances = sorted(list(enumerate(similarity[idx])),reverse=True,key=lambda x:x[1])
    
    songs = []
    for m_id in distances[1:21]:
        songs.append(df.iloc[m_id[0]].song)
        
    return songs


# In[95]:


recommendation('Heartbreak Hotel')


# In[96]:


import pickle
pickle.dump(similarity,open('similarity.pkl','wb'))
pickle.dump(df,open('df.pkl','wb'))


# In[ ]:




