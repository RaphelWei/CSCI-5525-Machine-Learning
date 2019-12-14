import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# initialize parameters
num_iter = 1000
lr = 1e-5
batch_size = 100

def shuffle_data(X, y):
    """
    shuffle training data to random order
    """
    m = X.shape[0]
    permutation = list(np.random.permutation(m))
    X_shuffled = X[permutation, :]
    y_shuffled = y[permutation]
    return X_shuffled, y_shuffled

def softmax(x):
    """
    softmax function
    """
    x -= np.max(x) 
    softmax = np.exp(x).T / np.sum(np.exp(x),axis=1).T
    return softmax.T

def encode_label(y):
    """
    encode the labels of data
    there are 10 classes in this case, 
    for example, input data with label 0,
    it will be encoded into
    [1 0 0 0 0 0 0 0 0 0]
    """
    n = len(y) 

    label = np.zeros((n, 10))
    for i in range(n):
        label[i,y[i]] = 1
    return label

def create_mini_batch(x,y,seed):
    """
    divide training data into a set of mini-batches
    """
    batch_x = x[seed*batch_size:(seed+1)*batch_size,:]
    batch_y = y[seed*batch_size:(seed+1)*batch_size]
    return batch_x, batch_y

def gradient_descent(w,x,y):
    """
    return loss function L  and gradient matrix
    """
    m = x.shape[0] 
    label = encode_label(y)
    scores = np.dot(x,w) 
    # change scores into probability with softmax
    prob = softmax(scores)
    # loss function
    loss = (-1 / m) * np.sum(label * np.log(prob)) 
    # gradient of loss function
    grad = (-1 / m) * np.dot(x.T,(label - prob))  
    return loss, grad

def multi_logistic_train(x,y,itr):
    """
    update w with gradient descent, using mini-batches as input
    """
    losses=[]
    m, d = x.shape
    k = len(np.unique(y))
    # initialize w
    w = np.random.rand(d, k)*0.00001
    num_batches = int(m/batch_size)

    for i in range(itr):
        seed = i%num_batches
        batch_x, batch_y = create_mini_batch(x,y,seed)
        loss,grad = gradient_descent(w,batch_x,batch_y)
        w = w - lr * grad
        losses.append(loss)
    
    return w, losses

def error_rate(y_predict, y):
    """
    compute the error rate between predicted labels and real labels
    """
    diff = y_predict-y
    return np.count_nonzero(diff)/len(diff)

def multi_logistic_predict(X, w):
    """
    predict labels of input data using
    """
    prob = softmax(np.dot(X,w))
    # find the largest probability, take this class as label of input data
    y_predict = np.argmax(prob, axis=1)
    return y_predict 

def main():
    # read data
    train_set = pd.read_csv("mnist_train.csv", header = None).values
    test_set = pd.read_csv("mnist_test.csv", header = None).values

    X_train = train_set[:,1:]
    y_train = train_set[:,0].reshape(-1,1)

    X_test = test_set[:,1:]
    y_test = test_set[:,0].reshape(-1,1)
    # insert X0, a vector with all 1s, to the first column of feature matrix X
    X0 = np.ones((X_train.shape[0], 1))
    X_train = np.hstack((X0, X_train))

    X0 = np.ones((X_test.shape[0],1))
    X_test = np.hstack((X0, X_test))

    X_shuffled, y_shuffled = shuffle_data(X_train, y_train)
    w, losses = multi_logistic_train(X_shuffled, y_shuffled, num_iter)

    np.save("weights.npy", w)
    y_predict = multi_logistic_predict(X_test, w).reshape(-1,1)
    accuracy = (1-error_rate(y_predict, y_test))*100
    print("The accuracy of test data is %f%%" % accuracy)

    plt.plot(losses)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()


if __name__ == "__main__":
    main()