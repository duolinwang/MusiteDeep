import os
import theano
import time
import numpy as np
import pandas as pd

import keras.layers.core as core
import keras.layers.convolutional as conv
import keras.models as models
import keras.utils.np_utils as kutils
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import merge
from keras.layers import pooling
from keras.models import Model
from keras.engine.topology import Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint,Callback
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization
from keras.regularizers import WeightRegularizer, l1, l2
from methods.attention import Attention,myFlatten
import copy



def MultiCNN(trainX, trainY,valX=None, valY=None,
             batch_size=1200, 
             nb_epoch=500,
             earlystop=None,transferlayer=1,weights=None,forkinas=False,compiletimes=0,
             compilemodels=None,predict=False):
    input_row     = trainX.shape[2]
    input_col     = trainX.shape[3]
    
    
    trainX_t=trainX;
    valX_t=valX;
    if(earlystop is not None): 
        early_stopping = EarlyStopping(monitor='val_loss', patience=earlystop)
        nb_epoch=10000;#set to a very big value since earlystop used
    
    trainX_t.shape=(trainX_t.shape[0],input_row,input_col)
    if(valX is not None):
        valX_t.shape=(valX_t.shape[0],input_row,input_col)
    
    if compiletimes==0:         
         input = Input(shape=(input_row,input_col))
         filtersize1=1
         filtersize2=9
         filtersize3=10
         filter1=200
         filter2=150
         filter3=200
         dropout1=0.75
         dropout2=0.75
         dropout4=0.75
         dropout5=0.75
         dropout6=0
         L1CNN=0
         nb_classes=2
         batch_size=1200
         actfun="relu"; 
         optimization='adam';
         attentionhidden_x=10
         attentionhidden_xr=8
         attention_reg_x=0.151948
         attention_reg_xr=2
         dense_size1=149
         dense_size2=8
         dropout_dense1=0.298224
         dropout_dense2=0
         
         input = Input(shape=(input_row,input_col))
         x = conv.Convolution1D(filter1, filtersize1,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(input) 
         x = Dropout(dropout1)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter2,filtersize2,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
         x = Dropout(dropout2)(x)
         x = Activation(actfun)(x)
         x = conv.Convolution1D(filter3,filtersize3,init='he_normal',W_regularizer= l1(L1CNN),border_mode="same")(x)
         x = Activation(actfun)(x)
         x_reshape=core.Reshape((x._keras_shape[2],x._keras_shape[1]))(x)
         
         x = Dropout(dropout4)(x)
         x_reshape=Dropout(dropout5)(x_reshape)
         
         decoder_x = Attention(hidden=attentionhidden_x,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_x)) # success  
         decoded_x=decoder_x(x)
         output_x = myFlatten(x._keras_shape[2])(decoded_x)
         
         decoder_xr = Attention(hidden=attentionhidden_xr,activation='linear',init='he_normal',W_regularizer=l1(attention_reg_xr))
         decoded_xr=decoder_xr(x_reshape)
         output_xr = myFlatten(x_reshape._keras_shape[2])(decoded_xr)
         
         output=merge([output_x,output_xr],mode='concat')
         output=Dropout(dropout6)(output)
         output=Dense(dense_size1,init='he_normal',activation='relu')(output)
         output=Dropout(dropout_dense1)(output)
         output=Dense(dense_size2,activation="relu",init='he_normal')(output)
         output=Dropout(dropout_dense2)(output)
         out=Dense(nb_classes,init='he_normal',activation='softmax')(output)
         cnn=Model(input,out)
         cnn.compile(loss='binary_crossentropy',optimizer=optimization,metrics=['accuracy'])
         
    else:
               cnn=compilemodels
    
    if(predict is False):
         if(weights is not None and compiletimes==0): #for the first time
            print "load weights:"+weights;
            if not forkinas:
                 cnn.load_weights(weights);
            else:
                 cnn2=copy.deepcopy(cnn)
                 cnn2.load_weights(weights);
                 for l in range((len(cnn2.layers)-transferlayer)): #the last cnn is not included
                    cnn.layers[l].set_weights(cnn2.layers[l].get_weights())
                    #cnn.layers[l].trainable= False  # for frozen layer
         
         if(valX is not None):
             if(earlystop is None):
               fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch,validation_data=(valX_t, valY))
             else:
               fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(valX_t, valY), callbacks=[early_stopping])
         else:
             fitHistory = cnn.fit(trainX_t, trainY, batch_size=batch_size, nb_epoch=nb_epoch)
    
    
    return cnn
