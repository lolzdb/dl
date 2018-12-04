import tensorflow as tf
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow.contrib.slim as slim


kernel = 32
training = True
#num = [6, 12, 24, 16]
num=[[2,64],[2,128],[2,256],[2,512]]
model_path='./model'
num_class=10
max_step=500000
save_step=500


def conv2d(data, filter, kenel, stride, scope, init=tf.contrib.layers.variance_scaling_initializer(),
           activation=tf.nn.selu, biase=True):
    def conv(data, stride, atenion, weight, biase, w, h, k_w, k_h, activation):
        data_i = []
        i = 0
        data = tf.expand_dims(data, axis=1)
        while i < h and i + k_h-1  < h:
            data_j = []
            j = 0
            while j < w and j + k_w-1 < w:
                # proportion = activation(
                #     tf.reduce_sum(tf.multiply(data[:, :, :, i:i + k_h, j:j + k_w], atenion), axis=[-3, -2, -1]))
                # proportion = tf.reshape(proportion, [-1, 1, 1, k_h, k_w])
                # data_j.append(activation(tf.add(
                #     tf.reduce_sum(tf.multiply(data[:, :, :, i:i + k_h, j:j + k_w],weight),
                #                   axis=[-1, -2, -3]), biase)))
                data_j.append(data[:, :, :, i:i + k_h, j:j + k_w])
                j += stride
            data_i.append(data_j)
            i += stride
        hl=len(data_i)
        wl=len(data_j)
        proportion=activation(tf.reduce_sum(tf.multiply(data_i,atenion), axis=[-3, -2, -1]))
        proportion = tf.reshape(proportion, [hl,wl, -1,1, 1, k_h, k_w])
        print('proportion',proportion.shape)
        data=activation(tf.add(tf.reduce_sum(tf.multiply(tf.multiply(data_i,proportion),weight),axis=[-3,-2,-1]),biase))
        print('data',data.shape)
        data_i = tf.transpose(data, [2, 3, 0, 1])
        return data_i

    def pad(data, k_w, k_h):
        ud = int((k_h - 1) / 2)
        lr = int((k_w - 1) / 2)
        pad = [[0, 0], [0, 0], [ud, ud], [lr, lr]]
        data = tf.pad(data, pad)
        return data

    shape = data.shape
    if biase == None:
        trainable = False
    else:
        trainable = True
    weight = tf.get_variable(scope + '/weight', [filter, shape[1], kenel[0], kenel[1]], tf.float32, init)
    biase = tf.get_variable(scope + '/biase', 1, tf.float32, initializer=tf.zeros_initializer(), trainable=trainable)
    atenion = tf.get_variable(scope + '/atenion', [kenel[0] * kenel[1], shape[1], kenel[0], kenel[1]], tf.float32, init)
    data = pad(data, kenel[1], kenel[0])
    shape = data.shape
    data = conv(data, stride, atenion, weight, biase, shape[-1], shape[-2], kenel[0], kenel[1], activation)
    if activation == None:
        return data
    return activation(data)

def conv2(net, scope):
    with slim.arg_scope([slim.conv2d]):
        with tf.variable_scope("dense_unit" + scope):
            net = slim.conv2d(net, kernel, [1, 1])
            net = slim.conv2d(net, kernel, [3, 3])
    return net

def block(net, num, scope):
    with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        scope=scope):
        net = conv2(net, num[0], scope + str(0))
        concat = net
        i = 0
        while i < num:
            net = conv2(net, scope + str(i))
            concat = tf.concat([concat, net])
            i += 1
    return concat

def transition(net, scope):
    with tf.variable_scope("dense_unit" + scope): net = slim.conv2d(net, kernel, [1, 1], activation_fn=tf.nn.selu)
    net = slim.avg_pool2d(net, [2, 2], stride=2, scope='pool1')
    return net

def resunit(input,fillter,scope):
    with tf.variable_scope("Rse_unit" + scope):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),activation_fn=tf.nn.selu):
            print(input.shape)
            output = tf.nn.selu(input)
            output = slim.conv2d(output, fillter, [1, 1])
            output = slim.conv2d(output, fillter, [3, 3])
            output = slim.conv2d(output, fillter*4, [3, 3],activation_fn=None)
            return output+input

def resunitj(input,fillter,scope,start=False):
    with tf.variable_scope("Rse_unit" + scope):
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            biases_initializer=tf.zeros_initializer(),activation_fn=tf.nn.selu,):
            output=tf.nn.selu(input)
            output=slim.conv2d(output,fillter,[1,1])
            if start==False:
                output = slim.conv2d(output, fillter, [3, 3], stride=2)
                output = slim.conv2d(output, fillter*4, [1, 1],activation_fn=None)
                shutcut = slim.conv2d(input, fillter * 4, [1, 1], stride=2, activation_fn=None)
            else:
                output = slim.conv2d(output, fillter, [3, 3])
                output = slim.conv2d(output, fillter *4, [1, 1], activation_fn=None)
                shutcut = slim.conv2d(input, fillter * 4, [1, 1], activation_fn=None,stride=2)
            return output+shutcut

def block(net,num,scope):
    scope="block" + scope
    with tf.variable_scope(scope):
        if scope!='block1':
            net = resunitj(net, num[1], scope+str(1))
        else:
            net = resunitj(net, num[1], scope+str(1))
        print(net.shape)
        for i in range(2,num[0]+1):
            net=resunit(net, num[1], scope+'_'+str(i))
        return net

def ResNet(net,num):
    with tf.variable_scope("RseNet50" ):
        net=slim.conv2d(net,64,[7,7],activation_fn=tf.nn.selu,stride=2)
        net=slim.max_pool2d(net,[3,3],stride=2)
        index=1
        for i in num:
            net=block(net,i,str(index))
            index += 1
        net = slim.avg_pool2d(net, [2, 2])
    return net

def TestNet(net):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu):
        net=slim.conv2d(net, 64, [7, 7], stride=2)
        print(net.shape)
        net=tf.transpose(net,[0,3,1,2])
        print(net.shape)
        # net =conv2d(net, 64, [7, 7],2,'block1')
        # print(net.shape)
        net =conv2d(net, 64, [3, 3],2,'block2')
        print(net.shape)
        net =conv2d(net, 64, [3, 3],2,'block3')
        print(net.shape)
        net = tf.transpose(net, [0, 2,3, 1])
        net = slim.avg_pool2d(net, [4, 4])
        print(net.shape)
    return net


#mnist = input_data.read_data_sets('minis', one_hot=True)

def loss(net, y):
    net = slim.fully_connected(tf.contrib.layers.flatten(net), 10, scope='fcn1',activation_fn=None)
    label=tf.cast(tf.reshape(tf.one_hot(y,num_class,1.0,0.0),shape=[-1,10]),tf.int64)
    y=tf.reshape(y,shape=[-1])
    loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=y))
    accurate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,1),tf.argmax(label,1)),tf.float32))
    return loss,accurate,tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


def testloss(net,y):
    net = slim.fully_connected(tf.contrib.layers.flatten(net), 10, scope='fcn1')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y))
    accurate = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(y, 1)), tf.float32))
    return loss, accurate, tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

class cifar10:
    def standar(self,img):
        min=np.min(img)
        max=np.max(img)
        mark=False
        result=0
        if max==min or max==0:
            mark=True
            result=(img - min) /1.0
        else:
            result=(img-min)/(max-min)
        return mark,result
    def loaddata(self,path):
        x=[]
        y=[]
        with open(path,'rb') as f:
            data=pickle.load(f,encoding='bytes')
            x.append(data[b'data'].reshape(-1, 3, 32,32).transpose(0,2,3,1).astype("float"))
            y.append(data[b'labels'])
        x=np.array(x).reshape(-1, 3, 32,32).transpose(0,2,3,1).astype("float")
        y=np.array(y).reshape(-1,1)
        X=[]
        Y=[]
        for i in range(0,x.shape[0]):
            img=x[i,:,:,:].reshape(32,32,3)
            c0,img[:, :, 0]=self.standar(img[:,:,0])
            c1,img[:, :, 1]=self.standar(img[:, :, 1])
            c2,img[:, :, 2]=self.standar(img[:, :, 2])
            if c0==True or c1==True or c2==True:
                continue
            else:
                X.append(img)
                Y.append(y[i])
        return np.array(X),np.array(Y).astype(np.int32)

    def __init__(self,tain_path,test_path,batch_size):
        self.batch_size=batch_size
        self.X=[]
        self.Y=[]
        self.TestX=[]
        self.TestY=[]
        self.index=1
        self.train_path=tain_path
        self.X,self.Y=self.loaddata(tain_path[0])
        self.buffer=list(range(0,len(self.X)))
        np.random.shuffle(self.buffer)
        self.test_buffer =[]
        self.test_path=test_path

    def trainbatch(self):
        if len(self.buffer)<self.batch_size:
            self.X, self.Y = self.loaddata(self.train_path[self.index])
            self.buffer = list(range(0, len(self.X)))
            self.index=(self.index+1)%len(self.train_path)
            np.random.shuffle(self.buffer)
        x=[]
        y=[]
        key=self.buffer[:self.batch_size]
        for i in key:
            self.buffer.remove(i)
            x.append(self.X[i])
            y.append(self.Y[i])
        return np.array(x),np.array(y)

    def gettest(self):
        if len(self.test_buffer)==0 and len(self.test_path)>0:
            self.TestX, self.TestY = self.loaddata(self.test_path)
            self.buffer = list(range(0, self.X.shape[0]))
            self.test_path=""
        i=self.test_buffer.pop(0)
        return self.TestX[i, :, :, :],self.TestY[i]
step=1
data=cifar10(['./data/1','./data/2','./data/3','./data/4','./data/5'],['./data/6'],64)
x=tf.placeholder(tf.float32,shape=[None,32,32,3],name='imgs')
y=tf.placeholder(tf.int32,shape=[None,1],name='labels')
net=TestNet(x)
loss,accurate,adma=loss(net,y)
print('start')
mnist = input_data.read_data_sets("minis/", one_hot=True)

avgl=0.
avga=0.
log_step=50
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    while step<max_step:
        img,label=data.trainbatch()
        # img,label=mnist.train.next_batch(64)
        # img=np.reshape(img,[-1,28,28,1])
        los,acc,_=sess.run([loss,accurate,adma],feed_dict={x:img,y:label})
        if step%log_step==0:
            print(step,' loss=',avgl/log_step,'  accurate=',avga/log_step)
            avgl=0.
            avga=0.
        else:
            avga+=acc
            avgl+=los
        step+=1