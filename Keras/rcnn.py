import nltk
nltk.download('punkt')
nltk.download('stopwords')

import sys
import os
import re
import numpy as np

from sklearn.utils import shuffle

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import tensorflow as tf
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import LSTM,Dense
from keras.layers import Concatenate,Conv1D
from keras.layers import Input
from keras import metrics

from keras.layers import Embedding,Lambda

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def load_data(path):
  
  train_corpus = []
  with open(path+'train_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(line[:-1])
  with open(path+'train_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      train_corpus.append(line[:-1])
      
  test_corpus = []
  with open(path+'test_pos.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(line[:-1])
  with open(path+'test_neg.txt','r',encoding='latin1') as f:
    for line in f.readlines():
      test_corpus.append(line[:-1])
      
  train_labels = np.zeros(25000)
  train_labels[0:12500] = 1
  
  test_labels = np.zeros(25000)
  test_labels[0:12500] = 1
  
  train_corpus,train_labels = shuffle(train_corpus,train_labels)
  test_corpus,test_labels = shuffle(test_corpus,test_labels)
  
  return train_corpus,test_corpus,train_labels,test_labels

path = '../Datasets/IMDB/'

train_corpus,test_corpus,train_labels,test_labels = load_data(path)

train_length = len(train_corpus)
test_length = len(test_corpus)

path = ''

def load_glove_vectors(path):
  glove_model = {}
  with open(path+'glove.42B.300d.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
      splitline = line.split()
      word = splitline[0]
      embedding = np.array([float(val) for val in splitline[1:]])
      glove_model[word] = embedding
  return glove_model

glove_model = load_glove_vectors(path)

maxwords = 25000

max_len = 30

def generate_vectors(Xtrain,Xtest,maxwords,max_len):
  tokenizer = Tokenizer(num_words=maxwords)
  tokenizer.fit_on_texts(Xtrain)
  
  train_seq = tokenizer.texts_to_sequences(Xtrain)
  train_vec = pad_sequences(train_seq,maxlen=max_len)
  
  test_seq = tokenizer.texts_to_sequences(Xtest)
  test_vec = pad_sequences(test_seq,maxlen=max_len)
  
  return train_vec,test_vec,tokenizer.word_index

train_vec,test_vec,wordindex = generate_vectors(train_corpus,test_corpus,maxwords,max_len)

def generate_embeddings(maxwords,wordindex):
  
  total_length = maxwords
  embedding_matrix = np.random.rand(total_length+1,300)
  
  count = 1
  
  for word, i in wordindex.items():
    embedding_vector = glove_model.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    if(count==total_length):
      break
    count+=1
    
  return embedding_matrix

total = min(maxwords,len(wordindex))

embedding_matrix = generate_embeddings(total,wordindex)

left_vectors_train = np.zeros((train_length,max_len))
left_vectors_train[1,:] = train_vec[1,:]

right_vectors_train = np.zeros((train_length,max_len))
right_vectors_train[:-1,:] = train_vec[:-1,:]

left_vectors_test = np.zeros((test_length,max_len))
left_vectors_test[1,:] = test_vec[1,:]

right_vectors_test = np.zeros((test_length,max_len))
right_vectors_test[:-1,:] = test_vec[:-1,:]

embeddingdim = 300

lstm_hidden = 50
dense_hidden = 100

def rcnn_model(lstm_hidden,dense_hidden):
  
  general_vector = Input(shape=(max_len,))
  left_vector = Input(shape=(max_len,))
  right_vector = Input(shape=(max_len,))
  
  embed = Embedding(maxwords+1,embeddingdim,weights=[embedding_matrix],trainable=False)
  
  general_out = embed(general_vector)
  
  left_context = embed(left_vector)
  right_context = embed(right_vector)
  
  left_context = LSTM(lstm_hidden,return_sequences=True)(left_context)
  right_context = LSTM(lstm_hidden,return_sequences=True,go_backwards=True)(right_context)
  
  right_context = Lambda(lambda x:K.reverse(right_context,axes=1),output_shape=(max_len,lstm_hidden))(right_context)
  
  total_input = Concatenate(axis=-1)([general_out,left_context,right_context])
  
  dense = Dense(dense_hidden,activation='tanh')(total_input)
  
  max_pool = Lambda(lambda x:K.max(dense,axis=1))(dense)
  
  final_out = Dense(1,activation='sigmoid')(max_pool)
  
  model = Model(inputs=[general_vector,left_vector,right_vector],outputs=final_out)
  
  return model

rcnn = rcnn_model(lstm_hidden,dense_hidden)

rcnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

rcnn.fit([train_vec,left_vectors_train,right_vectors_train],train_labels,batch_size=50,epochs=25)

rcnn_accuracy = rcnn.evaluate([test_vec,left_vectors_test,right_vectors_test],test_labels)[1]

print("Accuracy is {}".format(rcnn_accuracy*100))
