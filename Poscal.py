import cv2
import numpy as np
from skimage import measure
from weight_matrix import *
from split import *
font=cv2.FONT_HERSHEY_COMPLEX
def poscal(img):
    img=img[:,:,0]
    kernel=np.ones((6,1),np.uint8)
    im=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel)
    im_labels=measure.label(im,connectivity=1,neighbors=8)
    num=im_labels.max()
    if num==0:
       im_s=np.zeros((1,5))
    else:
       im_s=np.zeros((num,5))
       for i in range(num):
           temp=np.where(temp==(i+1))
           im_s[i,0]=max(index[0])
           im_s[i,1]=min(index[0])
           im_s[i,2]=max(index[1])
           im_s[i,3]=min(index[1])
           im_s[i,4]=len(index[0])
           return im_s,im

def main_test():
    import scipy.io
    font=cv2.FONT_HERSHEY_COMPLEX
    data=scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')
    u_seq_abnormal=data['u_seq_abnormal']
    data=scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')
    v_seq_abnormal=data['v_seq_abnormal']
    weight=Weight_matrix().get_weight_matrix()
    thisSplitter=Spliter()
    fg_img=cv2.imread('../ref_data/ab_fg_pics/108.bmp')
    for i,item in enumerate(realPos):
        cv2.rectangle(ab_img,(int(item[3]),int(item[1])),(int(i[1])-5),font,0.4,(255,255,0),1)
        cv2.imshow('img',fg_im)
        cv2.imshow('img',im)
        cv2.imshow('img',img)
        if cv2.waitkey(0) & 0xff==27:
           cv2.destoryAllWindows();

if__name__=='__main__':
   main.test()
                         

