#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics


# In[7]:


books= pd.read_csv('BooksDataSet.csv')
books.head()


# In[8]:


books=books[['book_id', 'book_name', 'genre', 'summary']]
books.head(3)


# In[9]:


sn.countplot(x= books['genre'], palette='plasma')
plt.xticks(rotation= 'vertical')


# In[10]:


books['summary'].iloc[1]


# In[11]:


## Cleaning the text

def cleantext(text):

    text=re.sub(r'[^\w\s]', '', text)

    test =re.sub("[^a-zA-Z]"," ",text)

    text=' '.join(text.split())

    text=text.lower()

    return text


# In[12]:


books['summary']=books['summary'].apply(lambda x:cleantext(x))
books['summary'].iloc[1]


# In[13]:


def showmostfrequentwords(text,no_of_words):
    
    allwords = ''.join([char for char in text])
    allwords = allwords.split()
    fdist = nltk.FreqDist(allwords)
    
    wordsdf = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})
    
    df = wordsdf.nlargest(columns="count",n = no_of_words)
    
    plt.figure(figsize=(7,5))
    ax = sn.barplot(data=df,x = 'count',y = 'word')
    ax.set(ylabel = 'Word')
    plt.show()
    
    return wordsdf
    
    
# 25 most frequent words

wordsdf = showmostfrequentwords(books['summary'],25)


# In[14]:


nltk.download('stopwords')


# In[15]:


#Removing Stop Words

from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

#removing the stopwords

def removestopwords(text):

    removedstopword= [word for word in text.split(' ') if word not in stop_words]
    return ' '.join(removedstopword)

books['summary'] = books['summary'].apply(lambda x:removestopwords(x))
books['summary'].iloc[1]


# ### LEMMATIZING

# In[17]:


nltk.download('wordnet')
nltk.data.path.append("corpora")
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()

def lematizing(sentence):
    stemsentence=""
    for word in sentence.split():
        stem=lemma.lemmatize(word)
        stemsentence +=stem
        stemsentence+= " "
    stemsentence=stemsentence.strip()
    return stemsentence

books['summary']=books['summary'].apply(lambda x:lematizing(x))


# In[19]:


books['summary'].iloc[1]


# ### Stemming

# In[20]:


from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

def stemming(sentence):
    stemmed=""
    for word in sentence.split():
        
        stem=stemmer.stem(word)
        stemmed+=stem
        stemmed+=" "
    stemmed= stemmed.strip()
    return stemmed

books['summary']=books['summary'].apply(lambda text:stemming(text))
books['summary'].iloc[1]
    


# In[21]:


freq_df = showmostfrequentwords(books['summary'],25)


# ### Encoding

# In[22]:


books_list=list(books['genre'].unique())

encode= [i for i in range(len(books_list))]
mapper=dict(zip(books_list,encode))
print(mapper)


# In[23]:


books['genre']=books['genre'].map(mapper)
books['genre'].unique()


# ### MODEL BUILDING

# In[24]:


count_vec=CountVectorizer(max_df=0.90,min_df=2,
                          max_features=1000,stop_words='english')

bagofword_vec= count_vec.fit_transform(books['summary'])
bagofword_vec


# In[25]:


y= books['genre']

X_train,X_test,y_train,y_test= train_test_split(bagofword_vec,y,test_size=0.2)


# ### support vector classifier

# In[26]:


svc=SVC()
svc.fit(X_train,y_train)
svc_pred= svc.predict(X_test)
print(metrics.accuracy_score(y_test,svc_pred))


# ### Multinomial Naive Bayes

# In[27]:


mb=MultinomialNB()
mb.fit(X_train,y_train)
mb_pred=mb.predict(X_test)
print(metrics.accuracy_score(y_test,mb_pred))


# ### Random Forest Classifier

# In[28]:


rf= RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred= rf.predict(X_test)
print(metrics.accuracy_score(y_test,rf_pred))


# ### As Non of the above model is performing well, Changing CountVectorizer to TF-IDF

# In[29]:


X_traintf,X_testtf,y_traintf,y_testtf =train_test_split(books['summary'],y,test_size=0.2,random_state=557)


# In[30]:


tfidf= TfidfVectorizer(max_df=0.8,max_features=10000)
Xtrain_tfidf= tfidf.fit_transform(X_traintf.values.astype('U'))
Xtest_tfidf= tfidf.transform(X_testtf.values.astype('U'))


# In[31]:


svc = SVC()
svc.fit(Xtrain_tfidf,y_traintf)
svccpred = svc.predict(Xtest_tfidf)
print(metrics.accuracy_score(y_testtf,svccpred))


# In[32]:


mb = MultinomialNB()
mb.fit(Xtrain_tfidf,y_train)
mbpred = mb.predict(Xtest_tfidf)
print(metrics.accuracy_score(y_test,mbpred))


# ### MODEL TESTING

# In[33]:


def test(text,model):
    text=cleantext(text)
    text=removestopwords(text)
    text=lematizing(text)
    text=stemming(text)

    text_vector=tfidf_vectorizer.transform([text])
    predicted=model.predict(text_vector)
    return predicted

ans =books['summary'].apply(lambda text:test(text,mb))


# In[ ]:


ans


# In[34]:


predicted=[]

for i in range(len(ans)):

    idx_val= ans[i][0]
    predicted.append(list(mapper.keys())[list(mapper.values()).index(idx_val)])


# In[35]:


new_map= dict([(value,key) for key,value in mapper.items()])
books['Actual Genre']=books['genre'].map(new_map)


# In[36]:


books['Predicted']=np.array(predicted)


# In[ ]:


books=books[['book_name','summary', 'Actual Genre','Predicted']]


# In[ ]:


books


# In[ ]:


import pickle
file= open("bookgenremodel.pkl",'wb')
pickle.dump(mb,file)
file.close()


# In[ ]:


file= open('tfidf.pkl','wb')
pickle.dump(tfidf,file)
file.close()


# In[ ]:




