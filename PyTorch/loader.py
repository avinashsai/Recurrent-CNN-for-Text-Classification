import sys
import os
import re
import numpy as np
from sklearn.utils import shuffle

import torch

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

	train_label = torch.from_numpy(train_labels)
	test_label = torch.from_numpy(test_labels)

	return train_corpus,test_corpus,train_label,test_label


def load_embeddings(path):
	glove_model = {}
	with open(path+'glove.42B.300d.txt','r',encoding='utf-8') as f:
		for eachline in f.readlines():
			words = eachline.split()
			word = words[0]
			vector = np.array([float(val) for val in words[1:]])
			glove_model[word] = vector

		return glove_model
