
# Getting Started
This is the Pytorch Implementation of Recurrent CNN for text classification

# Required Packages
1. Pytorch>=0.4

2. Glove Pre trained Vectors(27B)

# Usage

1. Download Glove pretrained vectors and mention corresponding path in ```main.py```

run the model using

```
python main.py -m (Vocabulary Size)
               -s (Maximum Sentence Length)
               -b (BatchSize)
               -l (Number of hidden LSTM cells)
               -d (Number of hidden units of Dense Layer)
               -o (Optimizer (Adam or SGD))
               -e (Number of Epochs)
```
               -
