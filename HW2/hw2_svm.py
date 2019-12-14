import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# reading data
df = pd.read_csv("hw2data.csv", header = None).values
np.random.shuffle(df)

N = df.shape[0]
len_training_set = int(0.8*N)

X0 = np.ones((N, 1))
df = np.concatenate((X0, df), axis = 1)

training_set = df[0:len_training_set,:]
test_set = df[len_training_set:,:]
X = training_set[:,0:-1]
y = training_set[:,-1].reshape(-1,1)
X_test = test_set[:,0:-1]
y_test = test_set[:,-1].reshape(-1,1)

C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

def cross_validation(X, y):
    X_shuffled = {}
    y_shuffled = {}
    num = len(y)
    index = np.arange(num)
    # shuffle the data
    np.random.shuffle(index)
    for i in range(10):
        X_shuffled[i] = X[index[i*num//10:(i+1)*num//10],:]
        y_shuffled[i] = y[index[i*num//10:(i+1)*num//10]].reshape(-1,1)
    return X_shuffled, y_shuffled

def get_next_train_valid(X_shuffled, y_shuffled, itr):
    X_valid = X_shuffled[itr]
    y_valid = y_shuffled[itr]
    X_train = merge_dict(itr, X_shuffled)
    y_train = merge_dict(itr, y_shuffled)
    return X_train, y_train, X_valid, y_valid

def merge_dict(itr, mydict):
    a = np.ones((1,mydict[itr].shape[1]))
    for i in range(10):
        if i != itr:
            a = np.concatenate((a, mydict[i]), axis=0)
    a = np.delete(a, 0, axis=0)
    return a

def svmfit(X_train, y_train, c):
    m,n = X_train.shape
    y = y_train.reshape(-1,1) * 1.
    X_dash = y * X_train
    H = np.dot(X_dash , X_dash.T) * 1.

    #Converting into cvxopt format - as previously
    P = matrix(H)
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    
    #Run solver
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    alphas = np.array(sol['x'])
    w = ((y * alphas).T @ X_train).reshape(-1,1)
    return w

def decision_boundary(a):
    for i in range(len(a)):
        if a[i]>=0.0:
            a[i]=1.0
        else:
            a[i]=-1.0

def predict(w,X_test):
    y_predict = X_test @ w
    return y_predict

def error_rate(y_predict, y):
    diff = y_predict-y
    return np.count_nonzero(diff)/len(diff)

def main():
    X_shuffled, y_shuffled = cross_validation(X, y)
    train_accuracy=[]
    valid_accuracy=[]
    test_accuracy =[]
    for c in C:
        train_error=[]
        valid_error=[]
        test_error =[]
        for itr in range(10):
            X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, itr)
            w = svmfit(X_train, y_train, c)
        
            # print train error rates
            y_train_predict = predict(w,X_train)
        
            decision_boundary(y_train_predict)
            error1 = error_rate(y_train_predict, y_train)
            # print("itr = %d, training error rate is %f" % (itr,error1))
            train_error.append(error1)
            # print valid error rates
            y_valid_predict = predict(w,X_valid)
            decision_boundary(y_valid_predict)
            error2 = error_rate(y_valid_predict, y_valid)
            # print("itr = %d, valid error rate is %f" % (itr,error2))
            valid_error.append(error2)
        
            # print test error rates
            y_test_predict = predict(w,X_test)
            decision_boundary(y_test_predict)
            error3 = error_rate(y_test_predict, y_test)
            # print("itr = %d, test error rate is %f" % (itr,error2))
            test_error.append(error3)
        # print("C = %f, the average training error rate is: %f" % (c, np.mean(train_error)))
        print("C = %f" % (c))
        train_accuracy.append(1-np.mean(train_error))
        valid_accuracy.append(1-np.mean(valid_error))
        test_accuracy.append(1-np.mean(test_error))

    print(train_accuracy)
    print(valid_accuracy) 
    print(test_accuracy)         
    plt.plot(train_accuracy, label = 'training set')
    plt.plot(valid_accuracy, label = 'validation set')
    plt.plot(test_accuracy, label = 'test set')          
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.xticks(np.arange(8), C)
    plt.legend()
    plt.show()

main()