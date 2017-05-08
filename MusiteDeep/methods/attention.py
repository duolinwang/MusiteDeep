from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec

class Attention(Layer):
    
    def __init__(self,hidden,init='glorot_uniform',activation='linear',W_regularizer=None,b_regularizer=None,W_constraint=None,**kwargs):
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.hidden=hidden
        super(Attention, self).__init__(**kwargs)
        
    def build(self, input_shape):
        #assert len(input_shape) == 2
        input_dim = input_shape[-1]
        self.input_length = input_shape[1]
        self.W0 = self.init((input_dim, self.hidden), name='{}_W3'.format(self.name))
        self.W = self.init((self.hidden, 1), name='{}_W'.format(self.name))
        self.b0 = K.zeros((self.hidden,), name='{}_b0'.format(self.name))
        self.b = K.zeros((1,), name='{}_b'.format(self.name))
        self.trainable_weights = [self.W0,self.W,self.b,self.b0]
        
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        
        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W0] = self.W_constraint
            self.constraints[self.W] = self.W_constraint
            
            
        super(Attention, self).build(input_shape)
        
    def call(self,x,mask=None):
            energy = self.activation(K.dot(x, self.W0)+self.b0)
            #energy=self.activation(K.dot(energy, self.W) + self.b)
            energy=K.dot(energy, self.W) + self.b
            energy = K.reshape(energy, (-1, self.input_length))
            energy = K.softmax(energy)
            xx = K.batch_dot(energy,x, axes=(1, 1))
            all=K.concatenate([xx,energy])
            return all
    
    
    def get_output_shape_for(self, input_shape):
        #assert input_shape and len(input_shape) == 2
        return (input_shape[0], input_shape[-1])#return (input_shape[0],input_shape[-1])
    
    def get_config(self):
        config = {'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'hidden': self.hidden.get_config() if self.hidden else None}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class myFlatten(Layer):
    def __init__(self,mydeletedim, **kwargs):
        #self.input_spec = [InputSpec(ndim='3+')]
        self.mydeletedim=mydeletedim
        super(myFlatten, self).__init__(**kwargs)
    
    def get_output_shape_for(self, input_shape):
        if not all(input_shape[1:]):
            raise Exception('The shape of the input to "Flatten" '
                            'is not fully defined '
                            '(got ' + str(input_shape[1:]) + '. '
                            'Make sure to pass a complete "input_shape" '
                            'or "batch_input_shape" argument to the first '
                            'layer in your model.')
        return (input_shape[0], np.prod(input_shape[1:]))
    
    def call(self, x, mask=None):
        x=x[:,:self.mydeletedim]
        return K.batch_flatten(x)
    