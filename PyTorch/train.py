import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_optimizer(optim,model):
	if(optim=='adam'):
		return torch.optim.Adam(model.parameters(),lr=0.001)
	elif(optim=='sgd'):
		return torch.optim.sgd(model.parameters(),lr=0.01)
	else:
		raise ValueError("Please Enter optimizers (adam,sgd)")

def get_accuracy(net,loader):

	correct = 0
	total = 0
	net.eval()
	for batchidx,(gen,left,right,labels) in enumerate(loader):
		labels = labels.double()
		pred = net(gen,left,right)
		pred = pred.view(pred.size(0))
		ypred = (pred>0.5).double()

		correct+=torch.sum(ypred==labels).item()
		total+=labels.size(0)

	return ((correct/total)*100)


def train_model(model,train_loader,test_loader,optim,epochs):
	optimizer = get_optimizer(optim,model)

	for epoch in range(epochs):
		model.train()
		for batchidx,(gen,left,right,labels) in enumerate(train_loader):
			labels = labels.view(-1,1).float()

			output = model(gen,left,right)
			loss = F.binary_cross_entropy(output,labels)

			model.zero_grad()
			loss.backward()
			optimizer.step()

		train_accuracy = get_accuracy(model,train_loader)
		print("Epoch {} Train Loss {} Train Accuracy {}".format(epoch,loss,train_accuracy))

	test_accuracy = get_accuracy(model,test_loader)
	print("Test Accuracy {}".format(test_accuracy))

