import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data


class rcnn(nn.Module):
	def __init__(self,embed_dim,lstm_hidden,dense_hidden,embed_matrix):
		super(rcnn,self).__init__()
		self.embed_dim = embed_dim
		self.lstm_hidden = lstm_hidden
		self.dense_hidden = dense_hidden
		self.embed_matrix = embed_matrix
		self.embed = nn.Embedding.from_pretrained(self.embed_matrix)
		self.embed.weight.requires_grad = False
		self.lstm = nn.LSTM(self.embed_dim,self.lstm_hidden,batch_first=True)
		self.dense1 = nn.Linear((300+2*self.lstm_hidden),self.dense_hidden)
		self.act1 = nn.Tanh()
		self.dense2 = nn.Linear(self.dense_hidden,1)
		self.act2 = nn.Sigmoid()

	def forward(self,gen,left,right):
		gen_out = self.embed(gen)
		left_out,_ = self.lstm(self.embed(left),None)
		right_out,_ = self.lstm(self.embed(right),None)

		right_out = torch.flip(right_out,[0])

		concat_out = torch.cat((gen_out,left_out,right_out),-1)
		final_out = self.act1(self.dense1(concat_out))

		final_out = torch.max(final_out,dim=1)[0]
		final_out = self.act2(self.dense2(final_out))

		return final_out
