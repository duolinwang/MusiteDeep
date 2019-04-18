from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import activations,initializers, regularizers, constraints
from keras.engine import InputSpec

class Attention(Layer):
    
    def __init__(self,hidden,init='glorot_uniform',activation='linear',W_regularizer=None,W0_regularizer=None,W_constraint=None,W0_constraint=None,**kwargs):
        super(Attention, self).__init__(**kwargs)
        self.init = initializers.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.W0_regularizer = regularizers.get(W0_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.W0_constraint = constraints.get(W0_constraint)
        self.hidden=hidden
        
        
    def build(self, input_shape):
        super(Attention, self).build(input_shape)
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.add_weight(shape=(input_dim, self.hidden), name='W0',initializer=self.init,regularizer=self.W0_regularizer,constraint=self.W0_constraint)
        self.W = self.add_weight(shape=(self.hidden, 1), name='W',initializer=self.init,regularizer=self.W_regularizer,constraint=self.W_constraint)
        self.b0 = self.add_weight(shape=(self.hidden,),name='b0',initializer=self.init)
        self.b = self.add_weight(shape=(1,),name='b',initializer=self.init)
        self.built = True
    
    def call(self,x,mask=None):
            super(Attention, self).call(x)
            energy = self.activation(K.dot(x, self.W0)+self.b0)
            #energy=self.activation(K.dot(energy, self.W) + self.b)
            energy=K.dot(energy, self.W) + self.b
            energy = K.reshape(energy, (-1, self.input_length))
            energy = K.softmax(energy)
            xx = K.batch_dot(energy,x, axes=(1, 1))
            all=K.concatenate([xx,energy])
            return all
    
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = {
            'hidden':self.hidden,
            'activation': activations.serialize(self.activation),
            'init': initializers.serialize(self.init),
            'W_regularizer': regularizers.serialize(self.W_regularizer),
            'W0_regularizer': regularizers.serialize(self.W0_regularizer),
            'W_constraint':constraints.serialize(self.W_constraint),
            'W0_constraint':constraints.serialize(self.W0_constraint)
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class myFlatten(Layer):
    def __init__(self,mydeletedim, **kwargs):
        super(myFlatten, self).__init__(**kwargs)
        self.mydeletedim=mydeletedim
    
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
    
    def call(self, x, mask=None):
        x=x[:,:self.mydeletedim]
        return K.batch_flatten(x)
    
    def get_config(self):
        config = {
            'mydeletedim':self.mydeletedim
        }
        base_config = super(myFlatten, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
