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

# initialize parameters
sigma = 0.1
c = 0.1

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

def rbf_kernel(x, y):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def rbf_svm_train(X_train, y_train, c):
    m,n = X_train.shape
    y = y_train.reshape(-1,1) * 1.
    K = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K[i,j] = rbf_kernel(X_train[i], X_train[j])

    P = matrix(np.outer(y_train,y_train) * K)

    #Converting into cvxopt format - as previously
    q = matrix(-np.ones((m, 1)))
    G = matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    h = matrix(np.hstack((np.zeros(m), np.ones(m) * c)))
    
    #Run solver
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    lamda = np.array(sol['x'])
    # w = ((y * alphas).T @ K).reshape(-1,1)
    return lamda.reshape(-1,1)

def decision_boundary(a):
    for i in range(len(a)):
        if a[i]>=0.0:
            a[i]=1.0
        else:
            a[i]=-1.0

def rbf_svm_predict(lamda,X_train,y_train,X_test):
    y_train.reshape(-1,1)
    num_test_data = X_test.shape[0]
    num_train_data = X_train.shape[0]
    K = np.zeros((num_test_data,num_train_data))
    for i in range(num_test_data):
        for j in range(num_train_data):
            K[i,j] = rbf_kernel(X_test[i], X_train[j])
    y_predict = K @ (lamda*y_train)
    return y_predict

def error_rate(y_predict, y):
    diff = y_predict-y
    return np.count_nonzero(diff)/len(diff)

def main():
    X_shuffled, y_shuffled = cross_validation(X, y)
    # train_error=[]
    valid_error=[]
    test_error =[]
    for itr in range(10):
        print(itr)
        X_train, y_train, X_valid, y_valid = get_next_train_valid(X_shuffled, y_shuffled, itr)
        lamda = rbf_svm_train(X_train, y_train, c)
        
        # print valid error rates
        y_valid_predict = rbf_svm_predict(lamda,X_train,y_train,X_valid)
        decision_boundary(y_valid_predict)
        error2 = error_rate(y_valid_predict, y_valid)
        print("fold = %d, valid error rate is %f" % (itr,error2))
        valid_error.append(error2)
        
        # print test error rates
        y_test_predict = rbf_svm_predict(lamda,X_train,y_train,X_test)
        decision_boundary(y_test_predict)
        error3 = error_rate(y_test_predict, y_test)
        print("fold = %d, test error rate is %f" % (itr,error3))
        test_error.append(error3)
    
    plt.plot(valid_error, label = 'validation set')
    plt.plot(test_error, label = 'test set')          
    plt.xlabel("fold")
    plt.ylabel("error rate")
    plt.xticks(np.arange(10))
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()