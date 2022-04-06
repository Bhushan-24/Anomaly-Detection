from weight_matrix import *
from Feature_extraction import *
from Classifiers import *
import scipy.io
import cv2
import matplotlib.pyplot as plt
def uvPlot(u,v,labels,timerSet=False):
    fig=plt.figure()
    for ind, label in enumerate(labels):
        if label:
           plt.scatter(u[ind],v[ind],c='red')
        else:
           plt.scatter(u[ind],v[ind],c='blue')
    if timerSet:
        timer=fig.canvas.new_timer(interval=1000)
        timer.add_callback(plt.close)
        plt.ylabel('v')
        plt.xlabel('u')
        plt.title('Optical Flow Features (U V) of training set \n Please close it to continue')
        plt.legend()
        plt.show()

def load_data():
    u_seq_abnormal=scipy.io.loadmat('../ref_data/u_seq_abnormal.mat')['u_seq_abnormal']
    v_seq_abnormal=scipy.io.loadmat('../ref_data/v_seq_abnormal.mat')['v_seq_abnormal']
    fg_imgs=['../ref_data/original_pics/' +str(i+1).zfill(3) + '.tif' for i in range(200)]
    original_imgs=['../ref_data/original_pics/' + str(i+1).zfill(3) + '/tif' for i in range(200)]
    abnormal_fg_imgs=['../ref_data/ab_fg_pics/' + str(i+1) + '.bmp for in range(200)]
    return u_seq_abnormal,v_seq_abnormal,fg_imgs,original_imgs,abnormal_fg_imgs

def plot(realPos,labels,img,classifier,timerSet=True):
    target=[None,0]
    for i, item in enumerate(realpos):
        if item[0]==item[1]:
           item[0]+=1;
        if item[2]==item[3]:
           item[2]+=1;
        score=item[-1]/((item[0]-item[1])*(item[2]-item[3]))
           if score>=target[-1]:
              target=[item,score]
    if target[-1]:
       item=target[0]
    cv2.rectangle(img,(int(item[3]),int(item[1])), (int(item[2]), int(item[0])),(0,0,255))
    if timeSet:
       if cv2.waitkey(100) & 0xff==27:
          cv2.destoryAllWindows()
def main():
    u_data,v_data,fg_imgs,original_imgs,abnormal_fg_imgs=load_data()
    weight=Weight_matrix().get_weight_matrix()
    thisFeaatureExtractor=Feature_extractor(original_imgs,fg_imgs,abnormal_fg_imgs,u_data,v_data,weight)
    train_data,train_labels=thusFeatureExtractor.get_features_and_labels(80,140)
    #uvPlot(train_data[:,0],train_data[:,1],train_labels,False)
    classifiers=Classifiers(train_data,train_labels)
    for name,model in classifiers.models.items():
        for ind,original_img in enumerate(original_imgs[:-1])
        pos,thisImg,_,_=thisFeatureExtractor.getPostition(fg_imgs,ind)
        features,_=thisFeatureExtractor.get_features_and_labels(ind,ind+1,False)
        labels=classifiers.models[name].predict(features)
        plot(pos,labels,thisImg,name)
    classifiers.prediction_metrics(test_data,test_labels,name)


if __name__=='__main__':
   

        
                  
    
              
