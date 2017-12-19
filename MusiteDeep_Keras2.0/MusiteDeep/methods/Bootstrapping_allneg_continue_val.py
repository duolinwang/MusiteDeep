#Bootstrapping_allneg_continue
from multiCNN import MultiCNN
from DProcess import convertRawToXY
import pandas as pd
import numpy as np
import keras.models as models
from keras.models import Model

def bootStrapping_allneg_continue_val(trainfile,valfile=None,srate=0.8,nb_epoch1=3,nb_epoch2=30,earlystop=None,maxneg=None,codingMode=0,transferlayer=1,inputweights=None,outputweights=None,forkinas=False): #inputfile:fragments (n*34);srate:selection rate for positive data;nclass:number of class models
  
  trainX = trainfile
  train_pos=trainX[np.where(trainX[:,0]==1)]
  train_neg=trainX[np.where(trainX[:,0]!=1)]
  train_pos=pd.DataFrame(train_pos)
  train_neg=pd.DataFrame(train_neg)
  if(train_pos.shape[0] == 0):
       print 'ERROR: size of positive sites is 0. Please check positive sites in training data!\n';
       exit()
  
  if(train_neg.shape[0] == 0):
       print 'ERROR: size of negative sites is 0. Please check negative sites in training data!\n';
       exit()
  
  train_pos_s=train_pos.sample(train_pos.shape[0]); #shuffle train pos
  train_neg_s=train_neg.sample(train_neg.shape[0]); #shuffle train neg
  slength=int(train_pos.shape[0]*srate);
  nclass=int(train_neg.shape[0]/slength);
  if(valfile is not None): 
     valX = valfile.as_matrix()
     val_pos=valX[np.where(valX[:,0]==1)]
     val_neg=valX[np.where(valX[:,0]!=1)]
     val_pos=pd.DataFrame(val_pos)
     val_neg=pd.DataFrame(val_neg)
     val_all=pd.concat([val_pos,val_neg])
     valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode) 
  else:
            a=int(train_pos.shape[0]*0.9);
            b=train_neg.shape[0]-int(train_pos.shape[0]*0.1);
            train_pos_s=train_pos[0:a]
            train_neg_s=train_neg[0:b];
            
            val_pos=train_pos[(a+1):];
            val_neg=train_neg[b+1:];
            
            val_all=pd.concat([val_pos,val_neg])
            valX1,valY1 = convertRawToXY(val_all.as_matrix(),codingMode=codingMode)
            slength=int(train_pos_s.shape[0]*srate); 
            nclass=int(train_neg_s.shape[0]/slength);
            
  if(maxneg is not None):
       nclass=min(maxneg,nclass); #cannot do more than maxneg times
  
  for I in range(nb_epoch1):
    train_neg_s=train_neg_s.sample(train_neg_s.shape[0]); #shuffle neg sample
    train_pos_ss=train_pos_s.sample(slength)
    for t in range(nclass):
        train_neg_ss=train_neg_s[(slength*t):(slength*t+slength)];
        train_all=pd.concat([train_pos_ss,train_neg_ss])
        trainX1,trainY1 = convertRawToXY(train_all.as_matrix(),codingMode=codingMode) 
        if t==0:
            models=MultiCNN(trainX1,trainY1,valX1,valY1,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t)
        else:
            models=MultiCNN(trainX1,trainY1,valX1,valY1,nb_epoch=nb_epoch2,earlystop=earlystop,transferlayer=transferlayer,weights=inputweights,forkinas=forkinas,compiletimes=t,compilemodels=models)
        
        print "modelweights assigned for "+str(t)+" bootstrap.\n";
        if(outputweights is not None):
            models.save_weights(outputweights,overwrite=True)
  
  
  return models;