import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
slim=tf.contrib.slim
data=tf.Variable(np.ones([1,4,4,4]))
weight=tf.Variable(np.ones([2,4,3,3]))

def conv(data,stride,atenion,weight,biase,w,h,k_w,k_h):
    data_i=[]
    i=0
    while i<h:
        data_j=[]
        j=0
        while j<w:
            proportion=slim.conv2d(data[:,:,:,i:i+k_h,j:j+k_w])
            data_j.append(tf.reduce_sum(tf.multiply(data[:,:,:,i:i+k_h,j:j+k_w],weight),axis=[-1,-2,-3]))
            j+=stride
        data_i.append(data_j)
        i+=stride
    data_i=tf.transpose(data_i,[2,3,0,1])
    return data_i

def pad(data,k_w,k_h):
    ud = int((k_h - 1) / 2)
    lr = int((k_w - 1) / 2)
    pad=[[0,0],[0,0],[ud,ud],[lr,lr]]
    data=tf.pad(data,pad)
    return data

def conv(data,kenel,filter,stride,scope,init=tf.contrib.layers.variance_scaling_initializer,activation=tf.nn.relu):
    shape=tf.shape(data)
    weight=tf.get_variable(scope+'/weight',[filter,shape[1],kenel[0],kenel[1]],tf.float32,init)
    biase=tf.get_variable(scope+'/biase',1,tf.float32,initializer=tf.zeros_initializer)
    atenion=tf.get_variable(scope+'/atenion',[filter,shape[1],kenel[0],kenel[1]],tf.float32,init)
    data=pad(data)
    shape = tf.shape(data)
    data=conv(data,stride,atenion,weight,biase,shape[-1],shape[-2],kenel[0],kenel[1])
    if activation==None:
        return data
    return activation(data)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    r=np.array(sess.run())
    print(r)