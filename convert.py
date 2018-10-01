import json
import numpy as np
import random
import cv2
import os
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf

def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def float_feature(value):
    """Wrapper for inserting float features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def convert_to_example(image_data, labels, labels_text, bboxes):
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    for b in bboxes:
        assert len(b) == 4
        # pylint: disable=expression-not-assigned
        [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
        # pylint: enable=expression-not-assigned
    difficult=np.zeros(len(labels),dtype=np.int64).tolist()
    truncated=np.zeros(len(labels),dtype=np.int64).tolist()
    shape=[1024,1024,3]
    image_format = b'JPEG'
    example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(shape[0]),
            'image/width': int64_feature(shape[1]),
            'image/channels': int64_feature(shape[2]),
            'image/shape': int64_feature(shape),
            'image/object/bbox/xmin': float_feature(xmin),
            'image/object/bbox/xmax': float_feature(xmax),
            'image/object/bbox/ymin': float_feature(ymin),
            'image/object/bbox/ymax': float_feature(ymax),
            'image/object/bbox/label': int64_feature(labels),
            'image/object/bbox/label_text': bytes_feature(labels_text),
            'image/object/bbox/difficult': int64_feature(difficult),
            'image/object/bbox/truncated': int64_feature(truncated),
            'image/format': bytes_feature(image_format),
            'image/encoded': bytes_feature(image_data)}))
    return example

def convert(datapath,dic,data,filelist,resultpath,sri,sj):
    batchsize=200
    recordsize=len(filelist)
    i=0
    ri=0
    j=sj
    while i<recordsize:
        tf_filename = resultpath+'/'+str(ri)+'.tfrecord'
        ri+=1
        with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
            while i<recordsize and j<batchsize:
                filepath=datapath+'/'+filelist[i]+'.jpg'
                img=Image.open(filepath)
                #nimg=img.tobitmap()
                nimg=img.tobytes()
                boxlist=[]
                lab=[]
                labtxt=[]
                for k in data[filelist[i]]['objects']:
                    if k['category'] in dic:
                        box=[int(k['bbox']['ymin']),int(k['bbox']['xmin']),int(k['bbox']['ymax']),int(k['bbox']['xmax'])]
                        boxlist.append(box)
                        lab.append(dic[k['category']])
                        labtxt.append(bytes(k['category'],encoding='utf-8'))
                example=convert_to_example(nimg, lab, labtxt, boxlist)
                tfrecord_writer.write(example.SerializeToString())
                i+=1
                j+=1
        if j==batchsize:
            j = 0
    return ri,j

def static(dic,ilist,id):
    for i in ilist:
        if i['category'] not in dic:
            ndic={}
            ndic['count']=0
            ndic['id']=[]
            dic[i['category']]=ndic
        dic[i['category']]['count']+=1
        dic[i['category']]['id'].append(id)

def getdic(cdic):
    j=0
    dic={}
    for i in cdic:
        dic[i]=j
        j+=1
    return dic

def getfilelist(path,dic,info):
    flist=[]
    result=[]
    for file in os.listdir(path):
        file=file.replace('.jpg','')
        flist.append(file)
    for i in flist:
        mark=0
        for k in data[i]['objects']:
            if k['category'] in dic:
                mark=1
                break
        if mark==1:
            result.append(i)
    print(len(flist),len(result))
    return result

def parsefile(jsonpath,imgspath):
    json_file = open(jsonpath, encoding='utf-8')
    list_file = open(imgspath, encoding='utf-8')
    imgs_list = json.load(list_file)
    data = json.load(json_file)
    cdic = {}
    for i in imgs_list:
        imgs = data['imgs'][i]
        static(cdic, imgs['objects'], i)
    json_file.close()
    list_file.close()
    return cdic, data

def getinfo(path):
    finfo = open(path)
    info=json.load(finfo)
    finfo.close()
    return info

def fileter(dic,low):
    for i in list(dic.keys()):
        if dic[i]['count']<low:
            dic.pop(i)

def swh(flist,data):
    h=[]
    w=[]
    for i in flist:
        for k in data[i]['objects']:
            w.append(k['bbox']['ymax']-k['bbox']['ymin'])
            h.append(k['bbox']['xmax']-k['bbox']['xmin'])
    return w,h

def mean(w,h,low,hight):
    count=0
    wm=0
    hm=0
    for i,j in zip(w,h):
        if i>low and i<hight:
            wm+=i
            hm+=j
            count+=1
    return wm/count,hm/count,count
def drawfill(box,img):
    draw = ImageDraw.Draw(img)
    draw.rectangle((box['xmin'],box['ymin'],box['xmax'],box['ymax']),fill=(0, 0, 0))


def clean(idspath,infopath,dic,datapath):
    idsfile=open(idspath)
    infofile=open(infopath)
    ids=json.load(idsfile)
    info=json.load(infofile)
    nids=[]
    ninfo={}
    for i in ids:
        object=[]
        mark=0
        count=0
        img=Image.open(datapath+'/'+i+'.jpg')
        for k in info[i]['objects']:
            if k['category'] in dic:
                object.append(k)
                count+=1
            else:
                mark=1
                drawfill(k['bbox'],img)
        if count != 0:
            infoitem = {}
            infoitem['id'] = i
            infoitem['objects'] = object
            ninfo[i]=infoitem
            nids.append(i)
            if mark !=0:
                img.save(datapath+'/'+i+'.jpg')
        img.close()
    idsfile.close()
    infofile.close()
    print(len(nids))
    print(len(nids))
    idsfile = open(idspath,'w')
    infofile = open(infopath,'w')
    idsjson=json.dumps(nids)
    infojson=json.dumps(ninfo)
    idsfile.write(idsjson)
    infofile.write(infojson)

class BatchRename():
    #  '''''
    #   批量重命名文件夹中的图片文件

    # '''
    def __init__(self,trainpath,infodir,idsdir):
        # 我的图片文件夹路径horse
        self.path = trainpath
        self.infodir=infodir
        self.idsdir=idsdir

    def rename(self):
        idsfile=open(self.idsdir+'/ids.json', encoding='utf-8')
        info_file = open(self.infodir+'/voc.json', encoding='utf-8')
        info=json.load(info_file)
        filelist=json.load(idsfile)
        voc={}
        ids=[]
        i = 10000
        n = 6
        for item in filelist:
            n = 6 - len(str(i))
            vname = str(0) * n + str(i)
            name = item
            img = {}
            img['id'] = vname
            img['objects'] = info[name]['objects']
            voc[vname] = img
            ids.append(vname)
            src = os.path.join(os.path.abspath(self.path), item+'.jpg')
            dst = os.path.join(os.path.abspath(self.path), vname + '.jpg')
            try:
                os.rename(src, dst)
                i = i + 1
            except:
                continue
        voc_json=json.dumps(voc)
        voc_file=open(self.infodir + '/' + 'voc.json','w')
        voc_file.write(voc_json)
        ids_file=open(self.infodir + '/' + 'ids.json','w')
        ids_json=json.dumps(ids)
        ids_file.write(ids_json)
        voc_file.close()
        ids_file.close()

def stastic(dic,idspath,infopath):
    idsfile=open(idspath)
    infofile=open(infopath)
    ids=json.load(idsfile)
    info=json.load(infofile)
    result={}
    for i in dic:
        item={}
        item['classc']=0
        item['imgc'] = 0
        item['count']=0
        result[i]=item
        print(i)
    for i in ids:
        for k in info['imgs'][i]['objects']:
            result[k['category']]['count']+=1
        for j in result:
            if result[j]['count']>0:
                result[j]['classc'] +=result[j]['count']
                result[j]['imgc'] += 1
                result[j]['count'] = 0
    return result


# enhancepath="F:\\ndata\\enhance"
# train="F:\\ndata\\train"
# einfo="F:\\ndata\\einfo.json"
# trinfo="F:\\ndata\\trinfo.json"
# resultpath="F:\\ndata\\record"
# info="F:\\ndata\\info.json"
# efl=getfilelist(enhancepath)
# trfl=getfilelist(train)
#
result,json_file=parsefile("F:\\data\\voc.json","F:\\data\\nids.json")
fileter(result,100)
# data=getinfo(info)
# dic=getdic(result)
# ifle=getfilelist(train,dic,data)
# data=[]
#ri,j=convert(train,dic,trdata,trfl,resultpath,0,1)
#ri,j=convert(enhancepath,dic,edata,efl,resultpath,ri,j)

voc="F:\\ndata\\voc.json"
ids="F:\\ndata\\ids.json"
path="F:\\ndata\\trainxml"
#stastic(result,ids,voc)
#r=BatchRename('F:\\ndata\\source','F:\\ndata','F:\\ndata')
