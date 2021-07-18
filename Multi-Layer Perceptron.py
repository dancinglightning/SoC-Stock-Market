# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:12:14 2021

@author: Hanan
"""

# Note that this is a 4-layer perceptron (1 input, 1 output & 2-hidden layers)

# Defining Functions

def sigmoid(x):      # sigmoid function
    e=2.718281828
    z=1/(1+(e)**(-x))
    return z

def sigmoid_diff(x):      # sigmoid differentiated function
    e=2.718281828
    z=-((e)**(-x))/((1+(e)**(-x))*(1+(e)**(-x)))
    return z

def signal(x):      # returns binary output for a function
    if x>=0:
        return 1
    else:
        return -1

def vector_multi(x,y):   # vector multiplication of two lists
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[x[i]*y[i]]
    return z1

def vector_add(x,y):     # vector addition of two lists
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[x[i]+y[i]]
    return z1
    
def vector_sub(x,y):     # vector subtraction of two lists
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[x[i]-y[i]]
    return z1
    
def vector_sigmoid(x):    # vector signal of a list
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[sigmoid(x[i])]
    return z1

def vector_sigmoid_diff(x):    # vector sigmoid differential of a list
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[sigmoid_diff(x[i])]
    return z1

def vector_scalar(k,x):    # vector scalar multiplaction
    z=len(x)
    z1=[]
    for i in range(z):
        z1=z1+[k*x[i]]
    return z1

def vector_error(x,y):
    t=0
    a=len(x)
    b=len(x[0])
    for i in range(a):
        for j in range(b):
            t+=(x[i][j]-y[i][j])*(x[i][j]-y[i][j])
    return t/(2*a)


# Main Program

w1=[]   # W(1)
w2=[]   # W(2)
w3=[]   # W(3)
y0=[[0.0611,0.2860,0.7464],[0.5102,0.7464,0.0860],[0.0004,0.6916,0.5006],[0.9430,0.4476,0.2648],[0.1399,0.1610,0.2477]]   # Inputs
y1=[]   # Y(1)
y2=[]   # Y(2)
y3=[]   # Outputs
d=[[0.4831],[0.5965],[0.5318],[0.6843],[0.2872]]    # Desired Outputs
n=0.3   # Learning rate
e=0     # Error range

cases=int(input("Enter number of testcases >>> "))
x0=int(input("Enter number parameters >>> "))
x1=int(input("Enter number of nodes in layer 1 >>> "))
x2=int(input("Enter number of nodes in layer 2 >>> "))
x3=int(input("Enter number outputs >>> "))

for i1 in range(x1):
    l=[]
    for j1 in range(x0):
        l+=[0]
    w1+=[l]
       
for i2 in range(x2):
    l=[]
    for j2 in range(x1):
        l+=[0]
    w2+=[l]
    
for i3 in range(x3):
    l=[]
    for j3 in range(x2):
        l+=[0]
    w3+=[l]
    
y1_inp=[]
y2_inp=[]
y3_inp=[]

for y1_rep in range(x1):
    y1_inp+=[0]
    
for y2_rep in range(x2):
    y2_inp+=[0]
    
for y3_rep in range(x3):
    y3_inp+=[0]
    
for case_rep in range(cases):
    y1+=[y1_inp]
    y2+=[y2_inp]
    y3+=[y3_inp]

print(y1)
print(y2)
print(y3)
    
epoch=0 
vector_error_previous=0   
while (vector_error(d,y3)-vector_error_previous)*(vector_error(d,y3)-vector_error_previous)>e*e:     
    
    vector_error_previous=vector_error(d,y3)   
    
    for case_rep_1 in range(cases):
        
        for p1 in range(x1):
            y1[case_rep_1][p1]=sum(vector_multi(w1[p1],y0[case_rep_1]))
        for p2 in range(x2):
            y2[case_rep_1][p2]=sum(vector_multi(w2[p2],y1[case_rep_1]))
        for p3 in range(x3):
            y3[case_rep_1][p3]=sum(vector_multi(w3[p3],y2[case_rep_1]))
    
        del3=[]
        del2=[]
        del1=[]
        
        print(w1)
        print(w2)
        print(w3)
        print(d)
        print(y3)
    
        for p4 in range(x3):
            del3+=vector_multi(vector_sub(d[case_rep_1],y3[case_rep_1]),(vector_sigmoid_diff(vector_multi(w3[p4],y2[case_rep_1]))))
            for p5 in range(x2):
                w3[p4][p5]+=n*(del3[p4])*y2[case_rep_1][p5]
            
                
        for p6 in range(x2):
            for p6x in range(x3):
                del2+=[del3[p6x]*w3[p6x][p6]]
            for p7 in range(x1):
                w2[p6][p7]+=-n*(del2[p6])*(sigmoid_diff(vector_multi(w2[p6],y1[case_rep_1])))*y1[case_rep_1][p7]
                
        for p8 in range(x1):
            for p8x in range(x2):
                del1+=[del2[p8x]*w2[p8x][p8]]
            for p9 in range(x0):
                w1[p8][p9]+=-n*(del1[p8])*(sigmoid_diff(vector_multi(w1[p8],y0[case_rep_1])))*y0[case_rep_1][p9]
    
        for p10 in range(x1):
            y1[case_rep_1][p10]=[vector_multi(w1[p10],y0[case_rep_1])]
        for p11 in range(x2):
            y2[case_rep_1][p11]=[vector_multi(w2[p11],y1[case_rep_1])]
        for p12 in range(x3):
            y3[case_rep_1][p12]=[vector_multi(w3[p12],y2[case_rep_1])]
    
    epoch+=1
    
print(w1)
print(w2)
print(w3)
    
