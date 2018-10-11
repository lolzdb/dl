import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class AnchorBox:
    def __init__(self,feature_size,anchor_size,feature_key,cell_size,img_size):
        self.feature_size=feature_size
        self.anchor_size=anchor_size
        self.feature_key=feature_key
        self.cell_size=cell_size
        self.img_size=img_size

    def getGride(self):
        result={}
        for i in self.feature_key:
            locate=self.getFeatureLocate(i)
            wh=self.getAncho(i)
            result[i]=tf.concat([locate,wh],axis=-1)
        print(result['0'])
        return result

    def getFeatureLocate(self,feature_key):
        w = list(range(0, self.feature_size[feature_key]['w']))
        h = list(range(0, self.feature_size[feature_key]['h']))
        x, y = tf.meshgrid(w, h)
        x=tf.cast(x,tf.float32)
        y = tf.cast(y, tf.float32)
        x *= self.cell_size[feature_key]
        y *= self.cell_size[feature_key]
        x += self.cell_size[feature_key] / 2
        y += self.cell_size[feature_key] / 2
        x=tf.expand_dims(x,axis=-1)
        y=tf.expand_dims(y,axis=-1)
        locate=tf.expand_dims(tf.concat([x,y],axis=-1),axis=-1)
        locate=tf.tile(locate,[1,1,1,len(self.anchor_size[feature_key])])
        print(locate.shape)
        return locate

    def getAncho(self,feature_key):
        anchor=np.array(self.anchor_size[feature_key]).astype('float')
        anchor[:,0]=anchor[:,0]/self.img_size[0]
        anchor[:, 1] = anchor[:,1] / self.img_size[1]
        anchor=tf.expand_dims(tf.expand_dims(tf.cast(anchor,tf.float32),axis=0),axis=0)
        anchor_size=tf.tile(anchor,[self.feature_size[feature_key]['h'],self.feature_size[feature_key]['w'],1,1])
        print(anchor_size.shape)
        return anchor_size




def getboxCode(anchor,box,label,threod):
    xmin=anchor[:,:,:,0]-anchor[:,:,:,2]/2
    ymin = anchor[:, :, :, 1] - anchor[:, :, :, 3] / 2
    xmax = anchor[:, :, :, 0] + anchor[:, :, :, 2] / 2
    ymax = anchor[:, :, :, 1] + anchor[:, :, :, 3] / 2
    anchor_arrea=(xmax-xmin)*(ymax-ymin)
    mark=tf.ones(xmin.shape,dtype=tf.float32)

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




feature_size={'0':{'w':8,'h':8}}
anchor_size={'0':[[4,8],[8,4]]}
cell_size={'0':8}
img_size=[64,64]
feature_key=['0']
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    a=AnchorBox(feature_size,anchor_size,feature_key,cell_size,img_size)
    result=a.getGride()
    gride=sess.run(result['0'])