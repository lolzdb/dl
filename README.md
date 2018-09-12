import tensorflow as tf

slim=tf.contrib.slim
kernel=32
training=True
def conv2(net,scope):
    with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.selu):
        with tf.variable_scope("dense_unit" + scope):
            net = slim.conv2d(net, kernel, [1, 1])
            net = slim.conv2d(net, kernel, [3, 3])
    return net

def block(net,num,scope):
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        scope=scope):
        net = conv2(net, num[0],scope+str(0))
        concat=net
        i=0
        while i <num:
            net=conv2(net,scope+str(i))
            concat=tf.concat([concat,net])
            i+=1
    return concat

def transition(net,scope):
    with tf.variable_scope("dense_unit" + scope):
        net = slim.conv2d(net, kernel, [1,1],activation_fn=tf.nn.selu)
        net = slim.avg_pool2d(net, [2, 2], stride=2,scope='pool1')
    return net

def DenseNet(net,num):
    i=0
    length=len(num)
    net = slim.conv2d(net, kernel, [7, 7])
    net = slim.conv2d(net, kernel, [3, 3],stride=2)
    while i<length-1:
        net=block(net, num[i], 'block'+str(i))
        net=transition(net,num[i])
        i+=1
    net = block(net, num[i], 'block' + str(i))
    net=slim.avg_pool2d(net, [2, 2])
    return net

num=[6,12,24,16]

def loss(net,y):
    net=slim.fully_connected(tf.flatten(net),10,scope='fcn1')
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net,labels=y)
    return tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
