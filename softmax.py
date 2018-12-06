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
    def conv(data, stride, atenion, weight, biase, k_w, k_h, activation,c):
        data=tf.image.extract_image_patches(data,[1,kenel[0],kenel[1],1],[1,stride,stride,1],padding='SAME',rates=[1,1,1,1])
        shape=data.shape
        data=tf.reshape(data,[-1,shape[1],shape[2],1,k_h*k_w,c])
        proportion=activation(tf.reduce_sum(tf.multiply(data,atenion),axis=[-2,-1]))
        proportion = tf.expand_dims(tf.expand_dims(proportion, axis=[3]),axis=-1)
        data=tf.multiply(proportion,data)
        data=tf.add(tf.reduce_sum(tf.multiply(data,weight),axis=[-1,-2]),biase)
        data=activation(data)
        return data

    shape = data.shape
    if biase == None:
        trainable = False
    else:
        trainable = True
    weight = tf.get_variable(scope + '/weight', [filter, kenel[0]*kenel[1],shape[-1]], tf.float32, init)
    biase = tf.get_variable(scope + '/biase', 1, tf.float32, initializer=tf.zeros_initializer(), trainable=trainable)
    atenion = tf.get_variable(scope + '/atenion', [kenel[0] * kenel[1], kenel[0]*kenel[1],shape[-1]], tf.float32, init)
    shape = data.shape
    data = conv(data, stride, atenion, weight, biase, kenel[0], kenel[1], activation,shape[-1])
    if activation == None:
        return data
    return activation(data)


def TestNet(net):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.selu):
        net=slim.conv2d(net, 64, [7, 7], stride=2)
        print(net.shape)
        # net =conv2d(net, 64, [7, 7],2,'block1')
        # print(net.shape)
        net =conv2d(net, 128, [3, 3],2,'block2')
        print(net.shape)
        net =conv2d(net, 256, [3, 3],2,'block3')
        print(net.shape)
        net = conv2d(net, 256, [3, 3], 2, 'block4')
        net = slim.avg_pool2d(net, [2, 2])
        print(net.shape)
    return net


#mnist = input_data.read_data_sets('minis', one_hot=True)

def loss(net, y,global_step,learning_rate):
    net = slim.fully_connected(tf.contrib.layers.flatten(net), 100, scope='fcn1',activation_fn=None)
    label=tf.cast(tf.reshape(tf.one_hot(y,100,1.0,0.0),shape=[-1,100]),tf.int64)
    y=tf.reshape(y,shape=[-1])
    loss =tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net, labels=y))
    accurate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,1),tf.argmax(label,1)),tf.float32))
    return loss,accurate,tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss,global_step= global_step)


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
            y.append(data[b'fine_labels'])
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
            self.TestX, self.TestY = self.loaddata(self.test_path[0])
            self.test_buffer = list(range(0, self.TestX.shape[0]))
            np.random.shuffle(self.test_buffer)
        x=[]
        y=[]
        key=self.test_buffer[:self.batch_size]
        for i in key:
            self.test_buffer.remove(i)
            x.append(self.TestX[i])
            y.append(self.TestY[i])
        return np.array(x),np.array(y)

step=1
globals_step=tf.Variable(0)
learning_rate = tf.train.exponential_decay(0.0002, globals_step, 800, 0.95, staircase = True)
data=cifar10(['./100/train'],['./100/test'],32)
x=tf.placeholder(tf.float32,shape=[32,32,32,3],name='imgs')
y=tf.placeholder(tf.int32,shape=[32,1],name='labels')
net=TestNet(x)
loss,accurate,adma=loss(net,y,globals_step,learning_rate)
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
        los,acc,rate,_=sess.run([loss,accurate,learning_rate,adma],feed_dict={x:img,y:label})
        if step%log_step==0:
            print(step,' loss=',avgl/log_step,'  accurate=',avga/log_step)
            avgl=0.
            avga=0.
        else:
            avga+=acc
            avgl+=los
        if step%10000==0:
            avg=0
            print('---------------start test----------------')
            for i in range(0,100):
                img, label=data.gettest()
                acc = sess.run([accurate], feed_dict={x: img, y: label})
                avg+=acc[0]
                if i%100==0:
                    print(acc)
            print('total-acc',avg/100)
        step+=1
        globals_step+=1