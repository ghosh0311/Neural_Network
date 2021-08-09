import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read data 
#The input layer has 784 nodes, corresponding to each of the 784 pixels 
#in the 28x28 input image. Each pixel has a value between 0 and 255, 
#with 0 being black and 255 being white

data = pd.read_csv("train.csv")

data = np.array(data)
m , n = data.shape
np.random.shuffle(data)
print("The dimension of MNIST dataset" , data.shape)

#The second layer has 10 nodes
#The output layer has 10 nodes
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev/255. #normalize

#Training sets
data_train = data[1000:10000].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255. #normalize
print("The dimension of training dataset:" ,data_train.shape)

data_test = data[10000:m].T
Y_test = data_train[0]
X_test = data_train[1:n]
X_test = X_test/255.
print("The dimension of test dataset:" ,data_test.shape)

#Initializing parameters
def init_params():
    W1 = np.random.rand(10 , 784) - 0.5 
    b1 = np.random.rand(10 , 1) - 0.5
    W2 = np.random.rand(10 , 10) - 0.5
    b2 = np.random.rand(10 , 1) - 0.5

    return W1 , b1 , W2 , b2

#Cost function
def costfunc(Y , h):
    m = Y.size
    cost = -(1/m)*(np.sum(np.multiply(np.log(h) , Y) + np.multiply((1-Y) , np.log(1-h))))
    return cost


#Sigmoid function
def sigmoid(z):
    g = 1.0 / (1.0 + np.exp(-z))
    return g

#Forward propagation
def fwd_prop(W1 , b1 , W2 , b2 , X , Y):
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = sigmoid(Z2)
    Ynew = transform(Y)
    loss = costfunc(Ynew , A2)

    return Z1 , A1 , Z2 , A2 , loss

#Encoding of prediction vector y
def transform(Y):
    Ynew = np.zeros((Y.size , Y.max() + 1))
    Ynew[np.arange(Y.size) , Y] = 1
    Ynew = Ynew.T

    return Ynew

def dsigmoid(Z):
    return Z > 0 


#Back Propagation
def back_prop(Z1 , A1 , Z2 , A2 , W2 , X , Y):
    m = Y.size
    Ynew = transform(Y)
    dZ2 = A2 - Ynew
    dW2 = (1/m)*dZ2.dot(A1.T)
    db2 = (1/m)*np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) + dsigmoid(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1)

    return dW1, db1, dW2, db2

#Updating parameters
def update_param(W1 , b1 , W2 , b2 , dW1 , dW2 , db1 , db2 , alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2

    return W1 , W2 , b1 , b2

#Predict
def get_predict(A2):
    return np.argmax(A2 , 0)

#Accuracy 
def accuracy(prediction , Y):
    return np.sum(prediction == Y)/Y.size

#Gradient Descent
def gradient_descent(X , Y , iteration , alpha):
    W1 , b1 , W2 , b2 = init_params()
    J_history = np.zeros((iteration, 1),dtype=float)
    for i in range(iteration):
        Z1 , A1 , Z2 , A2 , loss = fwd_prop(W1 , b1 , W2 , b2 , X , Y)
        print("Cost at iteration " ,i , "is:" , loss)
        dw1 , db1 , dw2 , db2 = back_prop( Z1 , A1 , Z2 , A2 , W2 , X , Y)
        W1 , W2 , b1 , b2 = update_param(W1 , b1 , W2 , b2 , dw1 , dw2 , db1 , db2 , alpha)
        J_history[i] = loss

    return W1 , b1 , W2 , b2 , J_history


#Setting alpha(learning rate) = 0.1
iter = 100
W1 , b1 , W2 , b2 , J = gradient_descent(X_train , Y_train , iter , 0.1)

#Testing the accuracy on test set 
Z1t , A1t , Z2t , A2t , loss = fwd_prop(W1 , b1 , W2 , b2 , X_test , Y_test)
#Printing the accuracy
print("Accuracy" , accuracy(get_predict(A2t) , Y_test)*100 , "%")


#Plot the loss curve
plt.plot(J , 'r')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Loss Curve")
plt.show()