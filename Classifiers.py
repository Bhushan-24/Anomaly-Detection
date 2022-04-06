from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_curve
import matplotlib.pyplot as plt
import numpy as np
class Classifiers(object):
      def __init__(self,train_data,train_labels,hyperTune=True):
          self.train_data=train_data
          self.train_labels=train_labels
          self.construct_all_models(hyperTune)
          
      def construct_all_models(self,hyperTune):
          self.models={'SVM':[SVC(kernel='linear',probability=True),dict(C=np.arange(0.0.1,2.01,0.2))],\ 'LogisticRegression':[lr(),dict(C=np.arange(0.1,3,0.1))],\ 'KNN' : (KNeighborsClassifier(),dict(n_neighbors=range(1,100))],)
          for name,candidate_hyperParam in self.models.itens():
              print('\n Training process finished\n\n\n')

      def train_with_hyperParanTuning(self,model_name,param_grid):
          grid=GridSearchCV(model,param_grid, cv=10, scroing='accuracy',n_jobs-1)
          grid.fit(self.train_data,self.train_labels)
          print('\n The best hyper-parameter for -- {} is {}, the corresponding mean accuracy through 10 Fold test is {} \n' \.format(name,auc))

          fpr,tpr,thresholds=roc_curve(test_labels.reshape(-1), prob[:,1],pos_label=1)
          self.roc_plot(fpr,tpr,name,auc)

      def roc_plot(self,fpr,tpr,name,auc):
          plt.figure(figsize=(20,5))
          plt.plot(fpr,tpr)
          plt.ylim([0.0,1.0])
          plt.ylim([0.0,1.0])
          plt.title('ROC of {} AUC: {} \n please close it to continue'.format(name,auc))
          plt.xlabel('False Positive Rate')
          plt.ylabel('True Positive Rate')
          plt.grid(True)
          plt.show()
