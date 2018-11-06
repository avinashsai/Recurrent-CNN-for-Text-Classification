import sys
import os
import re
import argparse


from loader import *
from generate import *
from model import *
from train import *


def main():

	embeddim = 300

	parser = argparse.ArgumentParser(description='Enter hyperparameters')

	parser.add_argument('-m','--maxwords',type=int,help='vocabulary size',required=True)
	parser.add_argument('-s','--sentencelength',type=int,help='Maximum sentence length',required=True)

	parser.add_argument('-b','--batchsize',type=int,help='Batch Size',required=True)
	parser.add_argument('-l','--lstmhidden',type=int,help='Lstm hidden',required=True)
	parser.add_argument('-d','--densehidden',type=int,help='dense hidden',required=True)
	parser.add_argument('-o','--optimizer',help='optimizer',type=str,required=True)
	parser.add_argument('-e','--epochs',type=int,help='epochs',required=True)

	args = vars(parser.parse_args())
	maxwords = args['maxwords']
	maxlen = args['sentencelength']
	batchsize = args['batchsize']
	lstmhidden = args['lstmhidden']
	densehidden = args['densehidden']
	optim = args['optimizer']
	epochs = args['epochs']
    

	data_path = '../Datasets/IMDB/'
	train_corpus,test_corpus,train_labels,test_labels = load_data(data_path)

	glove_path = ''
	glove_embeddings = load_embeddings(glove_path)

	train_vectors,test_vectors,embedding_matrix = generate_vectors(train_corpus,test_corpus,maxwords,glove_embeddings,maxlen)

	train_loader,test_loader = data_loader(train_vectors,train_labels,test_vectors,test_labels,batchsize)

	rcnn_model = rcnn(embeddim,lstmhidden,densehidden,embedding_matrix)

	print("Output of sample data: ",end=" ")

	gen = torch.randint(maxwords,(batchsize,embeddim)).long()
	left = gen
	right = gen
	out = rcnn_model(gen,left,right)
	print(out.shape)

	train_model(rcnn_model,train_loader,test_loader,optim,epochs)



if __name__ == '__main__':
	main()