import numpy as np
import matplotlib.pyplot as plt

### We will generate 500 samples in 3 Gaussian clouds
Nclass = 500

X1 = np.random.randn(Nclass,2) + np.array([0,-2]) ### Centered at (0,-2)
X2 = np.random.randn(Nclass,2) + np.array([2,2])  ### Centered at (2, 2)
X3 = np.random.randn(Nclass,2) + np.array([-2,2]) ### Centered at (-2,2)

X = np.concatenate((X1, X2 , X3))
print(X.shape)

### Labels
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass) ## Creates a vector of size (1500,) [0 0 0 0 ... 1 1 1 1 .... 2 2 2]
print(Y)
print(Y.shape)

### Visualizing the data
plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
plt.show()

### NN architecture parameters

D = 2 ### NUmber of features in our Data Set
M = 3 ### Hidden layer size
K = 3 ### Number of classes

### Random initialization of the weights
W1 = np.random.randn(D,M)
b1 = np.random.randn(M)
W2 = np.random.randn(M,K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    ### We apply the sigmoid activation function for the first hidden layer
    Z = 1/(1 + np.exp(-X.dot(W1) - b1))

    ### We apply the softmax activation function for the output layer as we did previously
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / (expA.sum(axis = 1, keepdims = True))
    return Y

def classification_rate(Y,P):
    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        n_total +=1
        if Y[i] == P[i]:
            n_correct+=1
    return float(n_correct)/n_total

#### Calling our FF function with the given parameters and calculate the classification rate:

P_Y_given_X = forward(X, W1, b1, W2, b2)
pred = np.argmax(P_Y_given_X, axis = 1)

### Assert that the length of our pred vector is equal to the Y vector
assert(len(pred) == len(Y))

### Lets now calculate the classification rate
### We expect low accuracy given that we haven't trained the NN at all:
print('Classification rate for randomly chosen weights: ', classification_rate(Y, pred))