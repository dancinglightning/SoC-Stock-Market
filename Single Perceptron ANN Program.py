# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 20:34:16 2021
@author: Hanan
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 00:24:59 2021
@author: Hanan
"""

def train_samples(i1,j1):   #function to create 2-D intented lists
    X=[]
    for i in range(i1):
        x=[]
        for j in range(j1):
            print("Enter value of ",j+1,"parameter of",i+1,"training sample",end=" ")
            z=eval(input(">>> "))
            x=x+[z]
            print()
        X=X+[x]
    return X

def sigmoid(x):      #sigmoid function
    z=1/(1+(2.71)**(-x))
    return z

def signal(x):      #returns binary output for a function
    if x>=0:
        return 1
    else:
        return -1

def vector_multi(x,y):   #vector multiplication of two lists
    z=len(x)
    z1=0
    for i in range(z):
        z1=z1+(x[i]*y[i])
    return z1

def vector_add(x,y):     #vector addition of two lists
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[x[i]+y[i]]
    return z1
    
def vector_sub(x,y):     #vector subtraction of two lists
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[x[i]-y[i]]
    return z1
    
def vector_signal(x):    #vector signal of a list
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[signal(x[i])]
    return z1

def vector_scalar(k,x):    #vector scalar multiplaction
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[k*x[i]]
    return z1

def error(w,X,d):
    z=[]
    for i in range(len(X)):
        z=z+[abs(d[i]-signal(vector_multi(w,X[i])))/2]
    return z
    

d=[]  #required output list
w=[]  #weight list
n=0.01  #learning rate
error_req=0  #required error

i1=int(input("Number of training samples >>> "))
j1=int(input("Number of training parameters >>> "))
error_req=float(input("Required error >>> "))

print()

X=train_samples(i1, j1)

for d_set in range(i1):
    print("Enter desired output of sample",d_set+1,end=" ")
    d1=eval(input(">>> "))
    d=d+[d1]

for w_set in range(j1+1):
    w=w+[0]
    
for X_set in range(i1):
    X[X_set]=X[X_set]+[-1]
    
epoch=1000
breaker=1


for epoch in range(epoch):
    for element in range(i1):
        if breaker==1:
            u=vector_multi(w, X[element])
            y=signal(u)
            w=vector_add(w, (vector_scalar(n*(d[element]-y),X[element])))
            if sum(error(w,X,d))/i1<=error_req:
                print("Req. weights :",w)
                breaker=0

print()


print("Time to test me !!!")
tests=int(input("Enter number of test runs >>> "))
print()

X1=train_samples(tests, j1)

for X1_set in range(tests):
    X1[X1_set]=X1[X1_set]+[-1]

A=[]
B=[]

for element1 in range(tests):
    u=vector_multi(w, X1[element1])
    y=signal(u)
    if y==1:
        A=A+[X1[element1]]
    else:
        B=B+[X1[element1]]

print("SET A :",A )
print("SET B :",B )

    
    
    