import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as mp
from PIL import Image
W1 = np.loadtxt("C:/Users/harik/Pictures/W1.txt")
W2 = np.loadtxt("C:/Users/harik/Pictures/W2.txt")
B1 = np.loadtxt("C:/Users/harik/Pictures/B1.txt")
B2 = np.loadtxt("C:/Users/harik/Pictures/B2.txt")
def get_image(image_path):
    image = Image.open(image_path).convert("L")
    pixel_values = list(image.getdata())
    return pixel_values

path = r'C:\Users\harik\Documents\mnist_test.csv'
data_l = pd.read_csv(path)
data = np.array(data_l)

m, n = data.shape
print(n)


def relu(x):
    return max(0.0, x)

def sig(x):
    fun = 1 / (1 + np.exp(-x))
    return fun

def d_sig(x):
    fun = x * (1 - x)
    return fun

def softmax(val,x):
    val = val/100000
    x = x / 100000
    return np.exp(val) / np.sum(np.exp(x))

def activation(W, B, X):
    out = W * X + B
    return np.sum(out)

def one_hot(y):
    enc = np.zeros(10)
    enc[y] = 1
    return enc.T,y

def der_soft(tot):
    der_tot = np.zeros(0)
    for a in range(len(tot)):
        b = (tot[a]) / np.sum((tot))
        der_s = b*(1-b)
        der_tot = np.append(der_tot,der_s)
    return der_tot


def der_relu(Z):
    return Z > 0


def back_prop(W1,B1,W2,B2,error,Z,l2out):
    a = der_soft(l2out)
    lr = 0.10
    dZ2 = error * a
    dZ1 = W2.T.dot(dZ2) * der_relu(Z)
    W2 = W2 + lr * dZ2
    B2 = B2 + lr * dZ2
    W1 = W1 + lr * dZ1
    B1 = B1 + lr * dZ1
    return W1,B1,W2,B2


def training(W1,B1,W2,B2,data):
    z = np.zeros(0)
    layer_2_array = np.zeros(0)
    layer_2_out = np.zeros(0)
    for i in range (500):
        out1 = activation(W1[0:784,0],B1[0],data[i,1:])
        out2 = activation(W1[0:784, 1], B1[1], data[i,1:])
        out3 = activation(W1[0:784, 2], B1[2], data[i, 1:])
        out4 = activation(W1[0:784, 3], B1[3], data[i, 1:])
        out5 = activation(W1[0:784, 4], B1[4], data[i, 1:])
        out6 = activation(W1[0:784, 5], B1[5], data[i, 1:])
        out7 = activation(W1[0:784, 6], B1[6], data[i, 1:])
        out8 = activation(W1[0:784, 7], B1[7], data[i, 1:])
        out9 = activation(W1[0:784, 8], B1[8], data[i, 1:])
        out10 = activation(W1[0:784, 9], B1[9], data[i,1:])
        act_out1 = relu(out1)
        act_out2 = relu(out2)
        act_out3 = relu(out3)
        act_out4 = relu(out4)
        act_out5 = relu(out5)
        act_out6 = relu(out6)
        act_out7 = relu(out7)
        act_out8 = relu(out8)
        act_out9 = relu(out9)
        act_out10 = relu(out10)    #print(out1,out2,out3,out4,out5,out6,out7,out8,out9,out10)
        z = np.append(z, (act_out1, act_out2, act_out3, act_out4, act_out5, act_out6, act_out7, act_out8, act_out9, act_out10))
        #print(layer_1_array)
        out1_layer2 = activation(W2[0:10,0],B2[0],act_out1)
        out2_layer2 = activation(W2[0:10, 1], B2[1], act_out2)
        out3_layer2 = activation(W2[0:10, 2], B2[2], act_out3)
        out4_layer2 = activation(W2[0:10, 3], B2[3], act_out4)
        out5_layer2 = activation(W2[0:10, 4], B2[4], act_out5)
        out6_layer2 = activation(W2[0:10, 5], B2[5], act_out6)
        out7_layer2 = activation(W2[0:10, 6], B2[6], act_out7)
        out8_layer2 = activation(W2[0:10, 7], B2[7], act_out8)
        out9_layer2 = activation(W2[0:10, 8], B2[8], act_out9)
        out10_layer2 = activation(W2[0:10, 9], B2[9], act_out10)
        layer_2_array = np.append(layer_2_array, (out1_layer2, out2_layer2, out3_layer2, out4_layer2, out5_layer2, out6_layer2, out7_layer2, out8_layer2, out9_layer2,out10_layer2))
        act_layer2_out1 = softmax(out1_layer2,layer_2_array)
        act_layer2_out2 = softmax(out2_layer2,layer_2_array)
        act_layer2_out3 = softmax(out3_layer2,layer_2_array)
        act_layer2_out4 = softmax(out4_layer2,layer_2_array)
        act_layer2_out5 = softmax(out5_layer2,layer_2_array)
        act_layer2_out6 = softmax(out6_layer2,layer_2_array)
        act_layer2_out7 = softmax(out7_layer2,layer_2_array)
        act_layer2_out8 = softmax(out8_layer2,layer_2_array)
        act_layer2_out9 = softmax(out9_layer2,layer_2_array)
        act_layer2_out10 = softmax(out10_layer2,layer_2_array)
        layer_2_out = np.append(layer_2_out,(act_layer2_out1,act_layer2_out2,act_layer2_out3,act_layer2_out4,act_layer2_out5,act_layer2_out6,act_layer2_out7,act_layer2_out8,act_layer2_out9,act_layer2_out10))
        t_ = np.max(layer_2_out)
        y,pos = one_hot(data[i,0])
        error  = y - layer_2_out
        #print(layer_2_out,error,"data", data[i,0])
        W1,B1,W2,B2 = back_prop(W1,B1,W2,B2,error,z,layer_2_out)
        z = []
        layer_2_array = []
        layer_2_out = []
    return W1,B1,W2,B2,t_

new_dat = get_image('C:/Users/harik/Pictures/a.png')
print(new_dat)

for j in range(1000):
    print(j)
    W1,B1,W2,B2,t_ = training(W1,B1,W2,B2,data)

np.savetxt("C:/Users/harik/Pictures/W1",W1)
np.savetxt("C:/Users/harik/Pictures/W2",W2)
np.savetxt("C:/Users/harik/Pictures/B1",B1)
np.savetxt("C:/Users/harik/Pictures/B2",B2)







