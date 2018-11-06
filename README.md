# Recurrent-CNN-for-Text-Classification

# Getting Started
This is the implementation of the paper https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552. This paper overcomes the drawback of fixed size window used in CNN by using RNN.It uses context of the word along with the embedding vector. Each word contains left context and right context except first and last words respectively. These context vectors are obtained by using Forward LSTM and backward LSTM. The outputs are concatenated with original word embedding on which Fully Connected Dense layer is applied. MaxPooling is used to obtain the vector that best represents the semantics.

# Implementation
I have used Glove 42B pretrained vectors instead of google pretrained vectors and adam is used as optimizer instead of SGD. Intialization is done is using glorot as opposed to uniform distribution mentioned in the paper.

Keras Folder - Implementation in Keras


PyTorch Folder - Implementation in PyTorch
