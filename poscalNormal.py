import cv2
import numpy as np
from skimage import measure
font=cv2.FONT_HERSHEY_COMPLEX
def poscalNormal(img1,img4):
    img1=img1[:,:,0]
    kernal=np.ones((6,1).np.uint8)
    im=cv2.morphologyEx(img1,cv2.MORPH_OPEN,kernal)
    im=cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel)
    im_labels=measure.label(im,connectivity=1,neighbors=8)
    num=im_labels.max()
    if num==0:
       im_s=np.zeros((1,5))
    else:
       im_s=np.zeros((num,5))
       for i in range(num):
           temp=np.copy(im_labels)
           temp[temp!=(i+1)]=0
           index=np.where(temp==(i+1))
           im_s[i,0]=max(index[0])
           im_s[i,1]=min(index[0])
           im_s[i,2]=max(index[1])
           im_s[i,3]=min(index[1])
           im_s[i,4]=len(index[0])
           return im_s,im

def main_test():
    img1=cv2.imread('../ref_data/fg_pics/169.bmp')
    img4=cv2.imread('../ref_data/ab_fg_pics/169.bmp')
    im_s,_=poscalNormal(img1,img4)
    print(im_s)
    img=cv2.imread('../ref_data/original_pics/169.tif')
    for i,item in enumerate(im_s):
        cv2.rectangle(img,(int(item[3]),int(item[1])),(int(item[2]),int(item[0])),(0,0,255))
        cv2.putText(img, str(i),(int(item[3]),int(item[1])-5),font,0.4,(255,255,0),1)
        cv2.imshow('img',img)
        if cv2.waitkey(0) & 0xff==27:
           cv2.destoryAllWindows()


if __name__=='__main__':
    main_test()
