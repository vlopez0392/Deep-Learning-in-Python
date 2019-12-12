### Logistic Regression with Softmax - Ecommerce_data
### This is an example of training a Logistic Regression classifier
### such that we can compare its performance with an ANN later on.

### Imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from preprocess import getData

#### The following function gets an indicator matrix from the targets
def y2indicator(y, K):
    N = len(y)
    ind = np.zeros((N,K))

    for i in range(N):
        ind[i, y[i]] = 1
    return ind

#### Getting our data, shuffling it and defining the test and train sets
X,Y = getData()
X,Y = shuffle(X,Y)
Y = Y.astype(np.int32)
D = X.shape[1]
K = len(set(Y))

X_train = X[:-100]
Y_train = Y[:-100]
Y_train_ind = y2indicator(Y_train,K)

X_test = X[-100:]
Y_test = Y[-100:]
Y_test_ind = y2indicator(Y_test,K)

#### Initializing the weights
W = np.random.randn(D,K)
b = np.zeros(K)

#### Defining a activation function, FPP, predict, classification and cost functions
def softmax(a):
    expA = np.exp(a)
    return expA/ expA.sum(axis =1, keepdims = True)

def forward(X,W,b):
    return softmax(X.dot(W)+b)

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y,P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

#### Training Logistic Regression with Softmax
train_costs = []
alpha = 0.001 ## Learning rate

for i in range(10000):
    pYtrain = forward(X_train, W, b)
    c_train = cross_entropy(Y_train_ind, pYtrain)
    train_costs.append(c_train)

    W -= alpha*X_train.T.dot(pYtrain-Y_train_ind)
    b -= alpha*(pYtrain-Y_train_ind).sum(axis = 0)

    if i%1000 == 0:
        print("Epoch: ", i, " Cost : ", c_train)

print("Final Classification Training Rate", classification_rate(Y_train, predict(pYtrain)))

#### Testing our classifier
pyTest = forward(X_test, W, b)
print("Final Classification Test Rate", classification_rate(Y_test, predict(pyTest)))

#### Plotting training costs
plt.plot(train_costs, label = 'train_cost')
plt.show()
