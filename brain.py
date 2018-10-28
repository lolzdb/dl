import tensorflow as tf
import numpy as np
import pickle
slim = tf.contrib.slim

def conv(net,filter,size,scope,stride=1):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.selu):
        with tf.variable_scope("unit" + scope):
            net=slim.conv2d(net, filter, size,stride=stride)
    return net

def front_groupconv(sub,input_num,group,scope,filter=8):
    index=0
    result=[]
    for i in sub:
        r=conv(i,filter,[3,3],scope+str(index))
        result.append(r)
        index += 1
    net=tf.concat(result,axis=3)
    return net

def back_groupconv(sub,input_num,group,fusion,stride,scope,filter=8):
    result=[]
    index=0
    for i in sub:
        output=tf.concat([i,fusion],axis=3)
        r=conv(output,filter,[3,3],scope+str(index),stride)
        result.append(r)
        index+=1
    net = tf.concat(result, axis=3)
    return net

def layer(net,input_num,group,filter,fusion_filter,scope,stride):
    sub = tf.split(net, group,axis=3)
    output=front_groupconv(sub,input_num,group,scope+'front',filter)
    fusion=conv(output,fusion_filter,[1,1],scope+'fusion')
    output = back_groupconv(sub, input_num, group, fusion,stride,scope+'back',filter)
    return output

def Brain(net):
    print(net.shape)
    net=conv(net, 64, [3,3], 'start',2) #16
    print(net.shape)
    net=layer(net, 64, 16, 8, 8, 'layer1', 2) #8
    net = layer(net, 128, 32, 8, 16, 'layer2', 2) #4
    net = layer(net, 256, 64, 8, 32, 'layer3', 2)  # 2
    net=slim.avg_pool2d(net, [2, 2])
    return net

def loss(net, y):
    net = slim.fully_connected(tf.contrib.layers.flatten(net), 10, scope='fcn1',activation_fn=None)
    label=tf.cast(tf.reshape(tf.one_hot(y,10,1.0,0.0),shape=[-1,10]),tf.int64)
    loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=label))
    accurate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,1),tf.argmax(label,1)),tf.float32))
    return loss,accurate,tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)

class cifar10:
    def standar(self,img):
        return (img-np.min(img))/(np.max(img)-np.min(img))

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
        for i in range(0,x.shape[0]):
            img=x[i,:,:,:].reshape(32,32,3)
            img[:,:,0]=self.standar(img[:,:,0])
            img[:, :, 1] = self.standar(img[:, :, 1])
            img[:, :, 2] = self.standar(img[:, :, 2])
            X.append(img)
        return np.array(X),np.array(y).astype(np.int32)

    def __init__(self,tain_path,test_path,batch_size):
        self.batch_size=batch_size
        self.X=[]
        self.Y=[]
        self.TestX=[]
        self.TestY=[]
        self.index=1
        self.train_path=tain_path
        self.X,self.Y=self.loaddata(tain_path[0])
        self.buffer=list(range(0,len(self.Y)))
        self.test_buffer =[]
        self.test_path=test_path

    def trainbatch(self):
        if len(self.buffer)<self.batch_size:
            self.X, self.Y = self.loaddata(self.train_path[self.index])
            self.buffer = list(range(0, self.X.shape[0]))
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
data=cifar10(['./data/1','./data/2','./data/3','./data/4','./data/5'],['./data/6'],16)
x=tf.placeholder(tf.float32,shape=[None,32,32,3],name='imgs')
y=tf.placeholder(tf.int32,shape=[None,1],name='labels')
net=Brain(x)
loss,accurate,adma=loss(net,y)
max_step=10000

with tf.Session() as sess:
    save = tf.train.Saver(max_to_keep=3)
    summary = tf.summary.FileWriter('./model', sess.graph)
    sess.run(tf.global_variables_initializer())
    while step<max_step:
        img,label=data.trainbatch()
        los,acc,log=sess.run([loss,accurate,adma],feed_dict={x:img,y:label})
        print(step,' loss=',los,'  accurate=',acc)
        step+=1