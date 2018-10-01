import json
import numpy as np
import random
import cv2
import os
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf

def getjson(infopath):
    infofile1=open(infopath)
    info1=json.load(infofile1)
    infofile1.close()
    return info1

def writejson(path,data):
    file = open(path,'w')
    jdata=json.dumps(data)
    file.write(jdata)
    file.close()

def fileter(idspath,infopath,datapath):
    idsfile=open(idspath,'r')
    infofile=open(infopath,'r')
    ids=json.load(idsfile)
    info=json.load(infofile)
    dlist=[]
    for i in ids:
        path=datapath+'/'+i+'.jpg'
        img=Image.open(path)
        w,h=img.size
        if w<1024 or h<1024:
           dlist.append(i)
        img.close()
    print(len(dlist))
    for i in dlist:
        ids.remove(i)
        info.pop(i)
    idsfile.close()
    infofile.close()
    idsfile = open(idspath, 'w')
    infofile = open(infopath, 'w')
    idsjson=json.dumps(ids)
    vocjson=json.dumps(info)
    idsfile.write(idsjson)
    infofile.write(vocjson)

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

def parsefile(jsonpath,imgspath):
    json_file = open(jsonpath, encoding='utf-8')
    list_file = open(imgspath, encoding='utf-8')
    imgs_list = list_file.readlines()
    data = json.load(json_file)
    cdic = {}
    length = len(imgs_list)
    i = 0
    while i < length:
        imgs_list[i] = imgs_list[i].replace('\n', '')
        i += 1
    for i in imgs_list:
        imgs = data['imgs'][i]
        static(cdic, imgs['objects'], i)
    json_file.close()
    list_file.close()
    return cdic,data

def delete(dic,low):
    for i in list(dic.keys()):
        if dic[i]['count']<low:
            dic.pop(i)

def static(idspath,infopath,datadir):
    ids=getjson(idspath)
    info=getjson(infopath)
    l=[]
    for i in ids:
        mark=0
        for k in info[i]['objects']:
            box = [int(k['bbox']['xmin'] ), int(k['bbox']['ymin'] ), int(k['bbox']['xmax'] ),
                   int(k['bbox']['ymax'] )]
            if box[0]<0 or box[1]<0 or box[2]>1024 or box[3]>1024:
                mark=1
        if mark==1:
            l.append(i)
    for i in l:
        ids.remove(i)
        info.pop(i)
    writejson(idspath,ids)
    writejson(infopath,info)
    return l

def clearids(idspath,infopath):
    ids = getjson(idspath)
    info = getjson(infopath)
    nids=[]
    for i in ids:
        if i in info:
            nids.append(i)
    writejson(idspath, nids)

def select(idspath,infopath,datapath,resultpath,up):
    ids = getjson(idspath)
    info = getjson(infopath)
    nids=[]
    ninfo={}
    for i in ids:
        object=[]
        count=0
        img=Image.open(datapath+'/'+i+'.jpg')
        for k in info[i]['objects']:
            w =int(k['bbox']['xmax'])-int(k['bbox']['xmin'])
            h= int(k['bbox']['ymax'])-int(k['bbox']['ymin'])
            if w>up and h>up:
                object.append(k)
                count+=1
            else:
                drawfill(k['bbox'],img)
        if count != 0:
            infoitem = {}
            infoitem['id'] = i
            infoitem['objects'] = object
            ninfo[i]=infoitem
            nids.append(i)
            img.save(resultpath+'\\'+i+'.jpg')
        img.close()
    print(len(ninfo))
    writejson(resultpath+'/ids.json', nids)
    writejson(resultpath+'/info.json', ninfo)

voc="F:\\ndata\\train3\\voc.json"
ids="F:\\ndata\\train3\\ids.json"
data='F:\\ndata\\train3\\source3'
result='F:\\ndata\\train3\\select'
up=12
select(ids,voc,data,result,up)