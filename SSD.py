import tensorflow as tf
import numpy as np

def encodeBoxs(cell_size,layer_shape,anchor_size):
    x=range(0,layer_shape['x'])
    y=range(0,layer_shape['y'])
    x,y=np.meshgrid(x,y)
    x=x*cell_size['x']+cell_size['x']/2
    y = y * cell_size['y'] + cell_size['y'] / 2
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
        boxencode.append(cell_size[i],layer_shape[i],anchor_size[i])
    return boxencode

def getGT(anchor,box,label,threod):
    xmin=anchor[:,:,:,0]-anchor[:,:,:,2]/2
    ymin = anchor[:, :, :, 1] - anchor[:, :, :, 3] / 2
    xmax = anchor[:, :, :, 0] + anchor[:, :, :, 2] / 2
    ymax = anchor[:, :, :, 1] + anchor[:, :, :, 3] / 2
    anchor_arrea=(xmax-xmin)*(ymax-ymin)
    mark=tf.ones(xmin.shape,dtype=tf.float32)
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
        return tf.less(i,len(label))[0]

    def body(i,sr,lr,rxmin,rymin,rxmax,rymax):
        score=jaryard(box[i])
        smask=tf.greater(score,threod)
        mask=tf.greater(score,sr)
        mask=tf.cast(mask,tf.int64)
        smask=tf.cast(smask,tf.int64)
        mask=tf.logical_and(smask,mask)
        lr=mask*label[i]+lr*(1-mask)
        sr=tf.where(mask,score,sr)
        rxmin=box[i][0]*mask+rxmin*(1-mask)
        rymin = box[i][1] * mask + rymin * (1 - mask)
        rxmax = box[i][2] * mask + rxmax * (1 - mask)
        rymax = box[i][3] * mask + rymax * (1 - mask)
        i+=1
        return [i,sr,lr,rxmin,rymin,rxmax,rymax]
    i=tf.constant(0,tf.int32)
    [i,sr,lr,rxmin,rymin,rxmax,rymax]=tf.while_loop([i,sr,lr,rxmin,rymin,rxmax,rymax])
    encode=tf.concat([rxmin,rymin,rxmax,rymax],axis=-1)
    return encode
