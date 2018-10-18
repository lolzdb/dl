import numpy as np
from PIL import Image
from PIL import ImageDraw
import copy
import json

width=1024
height=1024

def static(dic,ilist,id):
    for i in ilist:
        if i['category'] not in dic:
            ndic={}
            ndic['count']=0
            ndic['id']=[]
            dic[i['category']]=ndic
        dic[i['category']]['count']+=1
        dic[i['category']]['id'].append(id)

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

class Box:
    def __init__(self,box,index,type):
        self.xmin=box[0]
        self.ymin=box[1]
        self.xmax = box[2]
        self.ymax = box[3]
        self.type=type
        self.index=index
        self.x=float(box[2]+box[0])/2
        self.y=float(box[3]+box[1])/2

    def intersect(self,box):
        xmin=np.max([self.xmin,box.xmin])
        ymin = np.max([self.ymin, box.ymin])
        xmax = np.min([self.xmax, box.xmax])
        ymax = np.min([self.ymax, box.ymax])
        xr=(xmax-xmin)*(ymax-ymin)
        if xmax<=xmin or ymax<=ymin:
            return False
        return True

    def distance(self,box):
        return np.sqrt(np.square(self.x-box.x)+np.square(self.y-box.y))

    def getbox(self):
        return [self.xmin,self.ymin,self.xmax,self.ymax]

class Bmap:
    def __init__(self):
        self.xp=[]
        self.yp=[]
        self.box=Box([0,0,0,0],0,0)
        self.index=[]

    def add(self,box):
        self.xp.append(box)
        self.yp.append(box)
        self.index.append(box.index)

    def deletei(self,index):
        for i in self.xp:
            if i.index==index:
                target=i
                break
        self.xp.remove(target)
        for i in self.yp:
            if i.index==index:
                target=i
                break
        self.xp.remove(target)
        self.index.remove(index)

    def deleteo(self,box):
        self.xp.remove(box)
        self.yp.remove(box)
        self.index.remove(box.index)

    def inmap(self,index):
        return index in self.index

    def update(self):
        self.xp.sort(key=lambda x:x.xmin)
        self.yp.sort(key=lambda x: x.ymin)
        xmax=0
        for i in self.xp:
            if i.xmax>xmax:
                xmax=i.xmax
        ymax=0
        for i in self.yp:
            if i.ymax>ymax:
                ymax=i.ymax
        self.box.xmin=self.xp[0].xmin
        self.box.xmax = xmax
        self.box.ymin = self.yp[0].ymin
        self.box.ymax = ymax

    def cleave(self,bmap):
        for i in bmap.xp:
           if i.intersect(self.box):
               return True
        return False

    def cleaveb(self,box):
        return box.intersect(self.box)

    def isOut(self):
       if self.box.xmax-self.box.xmin<=width and \
               self.box.ymax-self.box.ymin<=height:
           return False
       return True

    def getAll(self):
        for i in self.xp:
            i.xmin=i.xmin-self.box.xmin
            i.ymin = i.ymin - self.box.ymin
            i.xmax = i.xmax - self.box.xmin
            i.ymax = i.ymax - self.box.ymin
        return self.xp

class Crop:
    def __init__(self):
        self.boxs=[]

    def remove(self,l,index):
        target=[]
        for i in l:
            if i.index==index.index:
                target.append(i)
        if len(target)>0:
            l.remove(target[0])

    def isin(self,l,index):
        target = []
        for i in l:
            if i.index == index.index:
                target.append(i)
        if len(target) > 0:
            return True
        return False

    def one(self,bmap):
        left=bmap.box.xmin
        up=bmap.box.ymin
        w=bmap.box.xmax-bmap.box.xmin
        h=bmap.box.ymax-bmap.box.ymin
        xmin=int(bmap.box.xmin-float(left)/(2048-w)*(width-w))
        ymin=int(bmap.box.ymin-float(up)/(2048-h)*(height-h))
        xmax=xmin+width
        ymax=ymin+height
        if xmax>2048:
            s=xmax-2048
            xmax=2048
            xmin-=s
        if ymax>2048:
            s=ymax-2048
            ymax=2048
            xmin-=s
        bmap.box.xmin=xmin
        bmap.box.xmax = xmax
        bmap.box.ymin = ymin
        bmap.box.ymax = ymax
        return bmap

    def selectone(self,box):
        spare=[]
        b=box.pop(0)
        for i in self.boxs:
            l=b.distance(i)
            if l!=0 and l<width:
                spare.append(copy.deepcopy(i))
        bmap=Bmap()
        bmap.add(b)
        bmap.update()
        for i in spare:
            bmap.add(i)
            bmap.update()
            if bmap.isOut():
                bmap.deleteo(i)
                bmap.update()
            else:
                if self.isin(box,i):
                    self.remove(box,i)
        reg=self.one(copy.deepcopy(bmap))
        db=self.getd(reg)
        return reg,db

    def getd(self,bmap):
        dl=[]
        for i in self.boxs:
            if self.isin(bmap.xp,i)==False and bmap.cleaveb(i):
                dl.append(copy.deepcopy(i))
        for i in dl:
            i.xmin-=float(bmap.box.xmin)
            i.ymin-=float(bmap.box.ymin)
            i.xmax-=float(bmap.box.xmin)
            i.ymax-=float(bmap.box.ymin)
            if i.xmin<0:
                i.xmin=0
            if i.ymin<0:
                i.ymin=0
            if i.xmax>width:
                i.xmax=width
            if i.ymax>height:
                i.ymax=height
        return dl

    def more(self):
        spare=copy.deepcopy(self.boxs)
        blist=[]
        dlist=[]
        while True:
            b,d=self.selectone(spare)
            blist.append(b)
            dlist.append(d)
            if len(spare)==0:
                break
        return blist,dlist

    def crop(self,boxs):
        self.boxs=boxs
        reg=[]
        dl=[]
        bmap=Bmap()
        for i in boxs:
            bmap.add(i)
        bmap.update()
        if bmap.isOut():
            r, d = self.more()
            reg.append(r)
            dl.append(d)
        else:
            reg.append([self.one(bmap)])
            dl.append([])
        return reg[0],dl[0]

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

def drawfill(box,img):
    draw = ImageDraw.Draw(img)
    draw.rectangle((box[0],box[1],box[2],box[3]),fill=(0, 0, 0))

def objectinfo(category,box):
    boxinfo = {'category': category, 'bbox': {'xmin':box[0],'ymin':box[1],'xmax':box[2],'ymax':box[3]}}
    return boxinfo

def crop(ids,info,datapath,resultpath):
    voc={}
    nids=[]
    cr=Crop()
    id=0
    for i in ids:
        file=datapath+'/'+i+'.jpg'
        img=Image.open(file)
        index=0
        boxs=[]
        for k in info[i]['objects']:
            box = [k['bbox']['xmin'], k['bbox']['ymin'], k['bbox']['xmax'], k['bbox']['ymax']]
            boxs.append(Box(box,index,k['category']))
            index+=1
        regin,dl=cr.crop(boxs)
        if len(regin)==1:
            image = {}
            n = 6 - len(str(id))
            name = str(0) * n + str(id)
            region = img.crop(regin[0].box.getbox())
            image['id'] = name
            image['objects'] = []
            rbox = regin[0].getAll()
            for b in rbox:
                image['objects'].append(objectinfo(b.type, b.getbox()))
            voc[name] = image
            region.save(resultpath + '/' + name + '.jpg')
            id += 1
            nids.append(name)
        else:
            for r, d in zip(regin, dl):
                image = {}
                n = 6 - len(str(id))
                name = str(0) * n + str(id)
                region = img.crop(r.box.getbox())
                for db in d:
                    drawfill(db.getbox(), region)
                image['id'] = name
                image['objects'] = []
                rbox = r.getAll()
                for b in rbox:
                    image['objects'].append(objectinfo(b.type, b.getbox()))
                voc[name] = image
                region.save(resultpath + '/' + name + '.jpg')
                id += 1
                nids.append(name)
    return voc,nids

# boxs=[Box([100,100,400,200],0,0),Box([410,100,710,200],1,0),Box([720,100,1030,200],2,0),Box([1040,100,1340,200],3,0),Box([1350,100,1650,200],4,0),Box([1660,100,1960,200],5,0)]
# cr=Crop()
# regin,dl=cr.crop(boxs)
# l=regin[0].getAll()
# d=dl
datapath='F:\\data\\test'
resultpath='F:\\data\\ctest'
result,json_file=parsefile("F:\\annotations.json","F:\\ids2")
ids=getjson('F:\\data\\test\\ids.json')
voc,ids=crop(ids,json_file['imgs'],datapath,resultpath)
info={'imgs':voc}
writejson(resultpath+'/nids.json',ids)
writejson(resultpath+'/voc.json',voc)