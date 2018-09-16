import tensorflow as tf
import pickle
import numpy as np

slim = tf.contrib.slim
kernel = 32
training = True
#num = [6, 12, 24, 16]
num=[[3,64],[4,128],[6,256],[3,512]]
model_path='./model'
num_class=10
max_step=10000
save_step=500

def conv2(net, scope):
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.selu):
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
        with slim.arg_scope([slim.conv2d],):
            print(input.shape)
            output= slim.conv2d(input,fillter,[1,1])
            output = slim.conv2d(output, fillter, [3, 3])
            output = slim.conv2d(output, fillter*4, [3, 3],activation_fn=None)
            return tf.nn.selu(output+input)

def resunitj(input,fillter,scope,start=False):
    with tf.variable_scope("Rse_unit" + scope):
        with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.selu):
            output=slim.conv2d(input,fillter,[1,1])
            if start==False:
                output = slim.conv2d(output, fillter*2, [3, 3], stride=2)
                output = slim.conv2d(output, fillter*4, [1, 1],activation_fn=None)
                shutcut = slim.conv2d(input, fillter * 4, [3, 3], stride=2, activation_fn=None)
            else:
                output = slim.conv2d(output, fillter, [3, 3])
                output = slim.conv2d(output, fillter *4, [1, 1], activation_fn=None)
                shutcut = slim.conv2d(input, fillter * 4, [1, 1], activation_fn=None)
            return tf.nn.selu(output+shutcut)

def block(net,num,scope):
    scope="block" + scope
    with tf.variable_scope(scope):
        if scope!='block1':
            net = resunitj(net, num[1], scope+str(1))
        else:
            net = resunitj(net, num[1], scope+str(1),True)
        print(net.shape)
        for i in range(2,num[0]+1):
            net=resunit(net, num[1], scope+'_'+str(i))
        return net

def ResNet(net,num):
    with tf.variable_scope("RseNet50" ):
        net=slim.conv2d(net,64,[7,7],activation_fn=tf.nn.selu,stride=2)
        index=1
        for i in num:
            net=block(net,i,str(index))
            index += 1
        net = slim.avg_pool2d(net, [2, 2])
    return net
def DenseNet(net, num):
    i = 0
    length = len(num)
    net = slim.conv2d(net, kernel, [7, 7])
    net = slim.conv2d(net, kernel, [3, 3], stride=2)
    while i < length - 1:
        net = block(net, num[i], 'block' + str(i))
        net = transition(net, num[i])
        i += 1
    net = block(net, num[i], 'block' + str(i))
    net = slim.avg_pool2d(net, [2, 2],stride=2)
    return net

def loss(net, y):
    net = slim.fully_connected(tf.contrib.layers.flatten(net), 10, scope='fcn1')
    label=tf.cast(tf.reshape(tf.one_hot(y,num_class,1.0,0.0),shape=[-1,10]),tf.int64)
    loss =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=label))
    accurate=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net,1),tf.argmax(label,1)),tf.float32))
    return loss,accurate,tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

class cifar10:
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
            X.append((img-np.min(img))/(np.max(img)-np.min(img)))
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
            self.buffer = range(0, self.TestX.shape[0])
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
data=cifar10(['./data/1','./data/2','./data/3','./data/4','./data/5'],['./data/6'],32)
x=tf.placeholder(tf.float32,shape=[None,32,32,3],name='imgs')
y=tf.placeholder(tf.int32,shape=[None,1],name='labels')
net=ResNet(x,num)
loss,accurate,adma=loss(net,y)

tf.summary.scalar('loss',loss)
tf.summary.scalar('accurate',accurate)
merge=tf.summary.merge_all()
with tf.Session() as sess:
    save = tf.train.Saver(max_to_keep=3)
    summary = tf.summary.FileWriter('./model', sess.graph)
    if len(model_path)==0:
        tf.train.Saver.restore(sess,tf.train.latest_checkpoint(model_path))
    sess.run(tf.global_variables_initializer())
    while step<max_step:
        img,label=data.trainbatch()
        los,acc,log,_=sess.run([loss,accurate,merge,adma],feed_dict={x:img,y:label})
        print(step,' loss=',los,'  accurate=',acc)
        summary.add_summary(log, step)
        if step%save_step==0:
            save.save(sess,model_path,global_step=step)
        step+=1