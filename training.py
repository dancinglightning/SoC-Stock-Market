import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x,0)

def log_err(x,y):
    return -y*np.log(x)-(1-y)*np.log(1-x)

def forward_prop(inp, w1, w2, w3, b1, b2, b3):
    a1 = relu(np.dot(w1.T, inp) + b1)
    a2 = sigmoid(np.dot(w2.T, a1) + b2)
    return {'a1': a1, 'a2': a2}

def backward_prop(inp, y, w1, w2, b1, b2, lr):
    dic = forward_prop(inp, w1, w2, b1, b2)
    dz2 = y - dic['a2']
    dw2 = np.dot(dic['a1'],dz2.T)
    db2 = dz2
    dz1 = np.dot(w2,dz2)*(dic['a1']>0)
    dw1 = np.dot(inp,dz1.T)
    db1 = dz1
    w1 -= lr*dw1
    w2 -= lr*dw2
    b1 -= lr*db1
    b2 -= lr*db2
    return (w1, w2, b1, b2)

def training(x, y, num_epochs, lr):
    w1 = np.random.randn(3,5)
    w2 = np.random.randn(5,2)
    b1 = np.random.randn(5,1)
    b2 = np.random.randn(2,1)
    for _ in range(num_epochs):
        err = 0
        for i in range(len(x)):
            (w1, w2, w3, b1, b2, b3) = backward_prop(np.reshape(x[i].T,(3,1)), y[i], w1, w2, b1, b2, lr)
            a2 = forward_prop(x[i], w1, w2, b1, b2)['a2']
            err += log_err(a2,y[i])
        print("EPOCH" + str(_) + "  :  " + str(err/len(x)))
    return (w1, w2, w3, b1, b2, b3)
