import tensorflow as tf
import numpy as np

def encodeBoxs(cell_size,layer_shape,anchor_size):
    x=range(0,layer_shape['x'])
    y=range(0,layer_shape['y'])
    x,y=np.meshgrid(x,y)
    x=x*cell_size['x']+cell_size['x']/2
    y = y * cell_size['y'] + cell_size['y'] / 2
    x=x[:,:,np.newaxis]
    y=y[:,:,np.newaxis]
    locate=np.zeros([layer_shape['x'],layer_shape['y'],len(anchor_size),4])
    locate[:,:,:,0]+=x
    locate[:,:,:,2]+=x
    locate[:, :, :, 1] += y
    locate[:, :, :, 3] += y
    for i in range(0,len(anchor_size)):
        locate[:,:,i,0]-=anchor_size[i]['w']/2
        locate[:, :, i, 2] += anchor_size[i]['w']/2
        locate[:, :, i, 1] -= anchor_size[i]['h']/2
        locate[:, :, i, 3] += anchor_size[i]['h']/2
    return locate

def getencode(cell_size,layer_shape,anchor_size,layer):
    boxencode=[]
    for i in layer:
        boxencode.append(encodeBoxs(cell_size[i],layer_shape[i],anchor_size[i]))
    return boxencode

def getGT(anchor,box,label,threod):
    xmin=anchor[:,:,:,0]
    ymin = anchor[:, :, :, 1]
    xmax = anchor[:, :, :, 2]
    ymax = anchor[:, :, :, 3]
    anchor_arrea=(xmax-xmin)*(ymax-ymin)
    rxmin=tf.zeros(xmin.shape)
    rymin=tf.zeros(xmin.shape)
    rxmax=tf.zeros(xmin.shape)
    rymax = tf.zeros(xmin.shape)
    sr=tf.zeros(xmin.shape)
    lr=tf.zeros(xmin.shape)

    def jaryard(bbox):
        uxmin=tf.maximum(bbox[0],xmin)
        uymin=tf.maximum(bbox[1],ymin)
        uxmax=tf.minimum(bbox[2],xmax)
        uymax = tf.minimum(bbox[3],ymax)
        nareea=(uxmax-uxmin)*(uymax-uymin)
        uarrea=anchor_arrea+(bbox[2]-bbox[0])*(bbox[3]-bbox[1])-nareea
        score=tf.div(nareea,uarrea)
        return score

    def condition(i,sr,lr,rxmin,rymin,rxmax,rymax):
        return tf.less(i,label.shape[0])

    def body(i,sr,lr,rxmin,rymin,rxmax,rymax):
        score=jaryard(box[i])
        smask=tf.greater(score,threod)
        mask=tf.greater(score,sr)
        mask=tf.cast(mask,tf.int64)
        smask=tf.cast(smask,tf.int64)
        mask=tf.logical_and(smask,mask)
        fmask=tf.cast(mask,tf.float32)
        lr=mask*label[i]+lr*(1-mask)
        mask = tf.cast(mask, tf.bool)
        sr=tf.where(mask,score,sr)
        rxmin=box[i][0]*fmask+rxmin*(1-fmask)
        rymin = box[i][1] * fmask + rymin * (1 - fmask)
        rxmax = box[i][2] * fmask + rxmax * (1 - fmask)
        rymax = box[i][3] * fmask + rymax * (1 - fmask)
        i+=1
        return [i,sr,lr,rxmin,rymin,rxmax,rymax]
    i=tf.constant(0,tf.int32)
    [i,sr,lr,rxmin,rymin,rxmax,rymax]=tf.while_loop([i,sr,lr,rxmin,rymin,rxmax,rymax])
    encode=tf.concat([rxmin,rymin,rxmax,rymax],axis=-1)
    return encode

def deconde(locate,anchor):
        pass

def filter_class(locate,predict,threosd=0.5):
    locate=tf.reshape(locate,[locate.shape[0],-1,locate.shape[-1]])
    predict=tf.reshape(predict,[predict.shape[0],-1,predict[-1]])
    lable=tf.argmax(predict,axis=-1)
    score=tf.reduce_max(predict,axis=-1)
    mask=tf.equal(lable,0)
    mask=tf.cast(mask,score.dtype)
    score=score*mask
    return locate,lable,score


def getclass(anchor,predict,locate,layer,threosd):
    score=[]
    lable=[]
    locates=[]
    for index in layer:
        lo,la,s=filter_class(locate,predict,threosd)
        score.append(s)
        locates.append(lo)
        lable.append(la)
    score=tf.concat(score,axis=1)
    lable=tf.concat(lable,axis=1)
    locate=tf.concat(locate,axis=1)
    return lable,score,locate

def getTop(lable, score, locate,topk):
    index=tf.nn.top_k(score,k=topk)
    score=tf.gather(score,index)
    lable=tf.gather(lable,index)
    locate=tf.gather(locate,index)
    return lable, score, locate

def pad_to_k(x,axis,k):
    rank =len(x.shape)
    npad = k - x.shape[axis]
    top = tf.stack([0]*rank)
    down = tf.stack([0]*axis+[npad]+[0]*(rank-axis-1))
    padding=tf.stack([top,down],axis=1)
    return tf.pad(x,padding)

def NMS(lable, score, locate,threosd,keep=100):
    index=tf.image.non_max_suppression(locate,score,threosd)
    score=tf.gather(score,index)
    lable = tf.gather(lable, index)
    locate=tf.gather(locate, index)
    score=pad_to_k(score,0,keep)
    lable=pad_to_k(lable,0,keep)
    locate = pad_to_k(locate, 0, keep)
    return lable, score, locate

def getNMS(lable, score, locate,threosd,keep=100):
    r=tf.map_fn(lambda x: NMS(x[0], x[1],x[2],threosd, keep),(lable, score, locate),dtype=(lable.dtype, score.dtype, locate.dtype),parallel_iterations=10,back_prop=False,swap_memory=False,infer_shape=True)
    lable, score, locate=r
    return lable, score, locate

def getdetect(anchor,predict,locate,layer,Lthreosd,Ithreosd,topk):
    locate = deconde(locate,anchor)
    lable, score, locate=getclass(predict,locate,layer,Lthreosd)
    lable, score, locate=getTop(lable, score, locate,topk)
    lable, score, locate=getNMS(lable, score, locate,Ithreosd)
    return lable, score, locate

cell_size={'0':{'x':8,'y':8}}
layer_shape={'0':{'x':8,'y':8}}
anchor_size={'0':[{'w':4,'h':4}]}
layer=['0']
box=tf.constant([[0.,0.,4.,4.],[12.,2.,14.,4.]])
lable=tf.constant([1,2])

with tf.Session() as sess:
    anchor=getencode(cell_size,layer_shape,anchor_size,layer)

