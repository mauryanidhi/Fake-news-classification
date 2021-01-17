# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 00:18:18 2020

@author: Nidhi
"""

from nltk.stem import PorterStemmer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px

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

messages=X.copy()

messages['title'][1]

messages.reset_index(inplace=True)

ps = PorterStemmer()

def process_title(title):
    new_title = title.lower()
    new_title = re.sub(r'\$[^\s]+', 'dollar', new_title)
    new_title = re.sub(r'[^a-z0-9\s]', '', new_title)
    new_title = re.sub(r'[0-9]+', 'number', new_title)
    new_title = new_title.split(" ")
    new_title = list(map(lambda x: ps.stem(x), new_title))
    new_title = list(map(lambda x: x.strip(), new_title))
    if '' in new_title:
        new_title.remove('')
    return new_title
titles = messages['title'].apply(process_title)

labels = np.array(df['label'])
# Get size of vocabulary
vocabulary = set()
for title in titles:
    for word in title:
        if word not in vocabulary:
            vocabulary.add(word)

vocab_length = len(vocabulary)

# Get max length of a sequence
max_seq_length = 0

for title in titles:
    if len(title) > max_seq_length:
        max_seq_length = len(title)

# Print results
print("Vocab length:", vocab_length)
print("Max sequence length:", max_seq_length)
#create the tokenizer and fit on title
tokenizer = Tokenizer(num_words=vocab_length)
tokenizer.fit_on_texts(titles)
#return index 
sequences = tokenizer.texts_to_sequences(titles)

word_index = tokenizer.word_index

model_inputs = pad_sequences(sequences, maxlen=max_seq_length)
model_inputs.shape
X_train, X_test, y_train, y_test = train_test_split(model_inputs, labels)
embedding_dim = 64


inputs = tf.keras.Input(shape=(max_seq_length,))

embedding = tf.keras.layers.Embedding(
    input_dim=vocab_length,
    output_dim=embedding_dim,
    input_length=max_seq_length
)(inputs)

gru = tf.keras.layers.GRU(units=embedding_dim)(embedding)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(gru)


model = tf.keras.Model(inputs, outputs)


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)


batch_size = 32
epochs = 3

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(),
        tf.keras.callbacks.ModelCheckpoint('model.h5', save_best_only=True)
    ]
)

fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'x': "Epoch", 'y': "Loss"},
    title="Loss Over Time"
)

fig.show()
fig = px.line(
    history.history,
    y=['auc', 'val_auc'],
    labels={'x': "Epoch", 'y': "AUC"},
    title="AUC Over Time"
)
