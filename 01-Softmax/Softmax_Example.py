##### Softmax example

##### This code exemplifies the use of Softmax function.
##### For simplicity, we pretend this is the output of a neural network or the activation at the last layer.

import numpy as np

#### Example 1:
#### Our activation will have 5 output nodes
a = np.random.randn(5)
print(a)

#### Applying softmax by exponentiation and division of sum
expa = np.exp(a)
expa = expa/(np.sum(expa))
print(expa)

### Confirm these probabilities add up to one
print(np.sum(expa))

#### Example 2:
#### Consider an output of 5 nodes but considering 100 samples
A = np.random.randn(100,5)
expA = np.exp(A)
expA = expA / np.sum(expA, axis = 1, keepdims=True)

