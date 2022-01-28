import numpy as np

# sigmoid function 
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T
#initialize the weight with a random value from -1 to 1 
w1 = np.random.randn(3 , 4) #we have 4 neural network and 3 input
#initialize the layer 1 
a1 = nonlin(np.dot(X , w1))
w2 = np.random.randn(4 , 1) #we have 1 neural network and 4  input
a2 = nonlin(np.dot(a1 , w2)) #initialize the layer 2 the output layer 

#training our model to get output closer and closer to the  expected value
for i in range(50000):
    #foreward proppagation 
    a1 = nonlin(np.dot(X , w1) , False)
    a2 = nonlin(np.dot(a1 , w2) , False)
    #backword proppagation
    d_w2 = np.dot(a1.T , 2*(y-a2)*nonlin(a2 , True))
    d_w1 = np.dot(X.T,(np.dot(2*(y - a2) * nonlin(a2 , True), w2.T) * nonlin(a1 , True)))
    w1 = d_w1+w1
    w2 = d_w2+w2
#our exepected output after the training process
print(a2)
