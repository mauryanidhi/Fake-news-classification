# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:18:43 2021

@author: Nidhi
"""


#Fake News Classifier Using LSTM
#Dataset: https://www.kaggle.com/c/fake-news/data#


import pandas as pd
df=pd.read_csv('train.csv')
df.head()

#Drop Nan Values
df=df.dropna()

## Get the Independent Features

X=df.drop('label',axis=1)

## Get the Dependent features
y=df['label']
X.shape
y.shape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU

### Vocabulary size
voc_size=5000

messages=X.copy()

messages['title'][1]

messages.reset_index(inplace=True)

import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')


### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
#Onehot Representation
onehot_repr=[one_hot(words,voc_size)for words in corpus] 
onehot_repr
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]


##1. Creating LSTM model using (dropout)
from tensorflow.keras.layers import Dropout
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.4))
model.add(LSTM(2))
model.add(Dropout(0.4))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

len(embedded_docs),y.shape


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


X_final.shape,y_final.shape
#splitting onto train &test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
#Model Training

### Finally Training
model.fit(X_train,y_train,validation_split=0.25,epochs=10,batch_size=55)
y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)



##3. Creating model gru using (dropout)
from tensorflow.keras.layers import Dropout

embedding_vector_features=10
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Dropout(0.3))
model.add(GRU(2))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print(model.summary())

len(embedded_docs),y.shape


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


X_final.shape,y_final.shape
#splitting onto train &test data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_split=0.25,epochs=15,batch_size=55)
y_pred=model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)