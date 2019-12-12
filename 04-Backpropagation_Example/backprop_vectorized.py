##### Vectorized Backpropagation Example
#####Imports

import numpy as np
import matplotlib.pyplot as plt

#### Functions from FNN file

def forward(X, W1, b1, W2, b2):
    ### We apply the sigmoid activation function for the first hidden layer
    Z = 1/(1 + np.exp(-X.dot(W1) - b1))

    ### We apply the softmax activation function for the output layer as we did previously
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / (expA.sum(axis = 1, keepdims = True))
    return Y,Z

def classification_rate(Y,P):
    n_correct = 0
    n_total = 0

    for i in range(len(Y)):
        n_total +=1
        if Y[i] == P[i]:
            n_correct+=1
    return float(n_correct)/n_total

####Defining the cost function
def cost(T,Y):
    tot = T*np.log(Y)
    return tot.sum()

####Defining the derivative functions
def d_w2(Z,T,Y):
    N,K = T.shape
    M = Z.shape[1]

    d_w2 = Z.T.dot(Y-K)
    assert(d_w2.shape == (M,K))
    return d_w2

def d_b2(T,Y):
    return (T-Y).sum(axis = 0)

def d_w1(X,Z,T,Y,W2):
    N,D = X.shape
    M,K = W2.shape

    d_w1 = X.T.dot((T-Y).dot(W2.T)*(Z*(1-Z)))
    assert(d_w1.shape == (D,M))

    return d_w1

def d_b1(T,Y,W2,Z):
    return ((T-Y).dot(W2.T)*Z*(1-Z)).sum(axis = 0)

#### Lets define a main function
def main():
    ###Create the data
    Nclass = 500
    X1 = np.random.randn(Nclass,2) + np.array([0,-2]) ### Centered at (0,-2)
    X2 = np.random.randn(Nclass,2) + np.array([2,2])  ### Centered at (2, 2)
    X3 = np.random.randn(Nclass,2) + np.array([-2,2]) ### Centered at (-2,2)

    X = np.concatenate((X1, X2 , X3))

    ### NN architecture parameters
    D = 2  ### Number of features in our Data Set
    M = 3  ### Hidden layer size
    K = 3  ### Number of classes

    ### LabelS of our targets
    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass) ## Creates a vector of size (1500,) [0 0 0 0 ... 1 1 1 1 .... 2 2 2]

    ### Converting the targets into an indicator variable
    N = len(Y)
    T = np.zeros((N,K))
    for i in range(N):
        T[i, Y[i]] = 1

    ### Visualizing the data
    plt.scatter(X[:,0], X[:,1], c = Y, s = 100, alpha = 0.5)
    plt.show()

    ### Random initialization of the weights and other parameters
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    learning_rate = 100e-7
    costs = []
    epochs = range(100000)

    for epoch in epochs:
        output,hidden = forward(X, W1, b1, W2, b2)

        if epoch%100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis = 1)
            r = classification_rate(Y,P)
            print("cost: ", c, "classification_rate: ", r)
            costs.append(c)

        #### We will not perform GA (Gradient Ascent)
        W2 += learning_rate*d_w2(hidden, T, output)
        b2 += learning_rate*d_b2(T, output)
        W1 += learning_rate*d_w1(X, hidden, T, output, W2)
        b1 += learning_rate*d_b1(T, output, W2, hidden)

    fig  = plt.figure()
    plt_epochs = range(1000)
    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    axes.plot(plt_epochs, costs, 'g-')
    axes.set_xlabel('Epoch ')
    axes.set_ylabel('Cost ')
    axes.set_title(' Epoch vs. Cost')
    plt.show()

if __name__ ==  '__main__':
    main()
