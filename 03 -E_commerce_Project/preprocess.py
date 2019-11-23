##### E-commerce Project
##### Imports
import pandas as pd
import numpy as np

#### Data pre-processing
def getData():
    """
    This function gets the data from the e_commerce.csv file and pre-processes it as we saw in the lectures
    """
    df = pd.read_csv('/home/vick/Python Projects/machine_learning_examples/ann_logistic_extra/ecommerce_data.csv')
    data = df.values

    X = data[:,:-1]
    Y = data[:, -1]

    ####  We will normalize the numerical columns first with Standard Scaling
    #### The numerical columns are are N_products_viewed and visit_duration
    X[:,1] = (X[:,1] - X[:,1].mean()) / (X[:,1].std())
    X[:,2] = (X[:,2] - X[:,2].mean()) / (X[:,2].std())

    ### We will also one-hot encode the categorical columns
    N,D = X.shape
    one_hot = np.zeros([N,D+3])
    one_hot[:, :D-1] = X[:, :D-1]

    for j in range(N):
        t = int(df['time_of_day'].iloc[j])
        one_hot[j,t+D-1] = 1

    return one_hot, Y

#### For the logistic classification problem we will only consider the first two classes
###  Consider the following function
def getBinaryData():
    X,Y = getData()
    X2 =  X[Y<=1]
    Y2 =  Y[Y<=1]
    return X2,Y2
