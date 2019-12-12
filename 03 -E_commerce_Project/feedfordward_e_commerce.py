import numpy as np
from preprocess import getData

### In this file we are going to create a feedforward NN process
### to obtain some predictions in our e-commerce data

### We will consider the tanh activation function

### Getting the data
X,Y = getData()

### NN parameters
M = 5
D = X.shape[1]
K = len(set(Y))

## Randomly creating the weights
W1 = np.random.randn(D,M)
b1 = np.zeros(M)
W2 = np.random.randn(M,K)
b2 = np.zeros(K)

## Softmax activation function
def softmax(a):
    expA = np.exp(a)
    return expA/(expA.sum(axis = 1, keepdims = True))

## FeedForward NN
def forward(X,W1,b1,W2,b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_Given_X = forward(X,W1,b1,W2,b2)
predictions = np.argmax(P_Y_Given_X, axis = 1)

### Classification rate
def classification_rate(Y,P):
    return np.mean(Y == P)

print("Score: " , classification_rate(Y, predictions))