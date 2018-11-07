import os
import re
import numpy as np 

import collections
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data


def get_vectors(vocabulary,corpus,maxlen):

	length = len(corpus)
	vectors = torch.zeros(length,maxlen).long()

	for i in range(length):
		lexicons = corpus[i].split()
		vector = torch.zeros(maxlen).long()
		count = 0
		for j in range(len(lexicons)):
			if lexicons[j] in vocabulary:
				vector[count] = vocabulary[lexicons[j]]
				count+=1
			if(count>=maxlen-1):
				vectors[i,:] = vector
				break
		vectors[i,:] = vector

	return vectors



def generate_embed_matrix(glove_vectors,maxwords,vocabulary):

	embedding_matrix = torch.rand(maxwords+1,300)
	i = 1
	for word in vocabulary:
		if(word in glove_vectors):
			embedding_matrix[i,:] = torch.from_numpy(glove_vectors[word])
		i+=1

	return embedding_matrix

def get_vocabulary(train_corpus,maxwords):
	words = []
	for sentence in train_corpus:
		words+=sentence.split()

	vocab = Counter(words).most_common(maxwords)

	index = 1
	vocabulary = {}
	for word,_ in vocab:
		vocabulary[word] = index
		index+=1

	return vocabulary


def generate_vectors(train_corpus,test_corpus,maxwords,glove_vectors,maxlen):

	vocabulary = get_vocabulary(train_corpus,maxwords)

	embedding_matrix = generate_embed_matrix(glove_vectors,maxwords,vocabulary)

	train_vectors = get_vectors(vocabulary,train_corpus,maxlen)

	test_vectors = get_vectors(vocabulary,test_corpus,maxlen)

	return train_vectors,test_vectors,embedding_matrix


def data_loader(train_vectors,train_labels,test_vectors,test_labels,batchsize):

	train_length = train_vectors.size(0)
	test_length = test_vectors.size(0)
	length = train_vectors.size(1)

	train_left = torch.zeros(train_length,length).long()
	train_left[1:,] = train_vectors[1:,]

	train_right = torch.zeros(train_length,length).long()
	train_right[:-1,] = train_vectors[:-1,]

	test_left = torch.zeros(test_length,length).long()
	test_left[1:,] = test_vectors[1:,]

	test_right = torch.zeros(test_length,length).long()
	test_right[:-1,] = test_vectors[:-1,]

	train_right_flip = torch.flip(train_right,[0])
	test_right_flip = torch.flip(test_right,[0])

	train_l = torch.utils.data.TensorDataset(train_vectors,train_left,train_right_flip,train_labels)
	train_loader = torch.utils.data.DataLoader(train_l,batch_size=batchsize)

	test_l = torch.utils.data.TensorDataset(test_vectors,test_left,test_right_flip,test_labels)
	test_loader = torch.utils.data.DataLoader(test_l,batch_size=batchsize)

	return train_loader,test_loader