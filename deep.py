import numpy as np
import tensorflow as tf
import keras
from keras.layers import Concatenate,Dense,BatchNormalization
from keras import Input, Model

def make_mlp(data):
    params=data.params_dict()
    model=single_builder(params=params)
    return MLP(params=params,
               model=model,
               class_weight=None)

def make_balanced_mlp(data):
    params=data.params_dict()
    model=single_builder(params=params)
    return MLP(params=params,
               model=model,
               class_weight=data.weight_dict())

class MLP(object):
    def __init__( self,
                  params,
                  model,
                  class_weight=None):
        self.params=params
        self.model=model
        self.class_weight=class_weight

    def fit(self,X,y):
        y=tf.one_hot(y,depth=self.params['n_cats'])
        return self.model.fit(x=X,
                              y=y,
                              class_weight=self.class_weight,
                              epochs=1000,
                              callbacks=basic_callback(),
                              verbose=False)
    def predict(self,X):
        y=self.model.predict(X,
                             verbose=False)
        return np.argmax(y,axis=1)
    
    def predict_proba(self,X):
        return self.model.predict(X,
                             verbose=False)

def basic_callback():
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)


def single_builder(params,
                   hyper_params=None,
                   class_dict=None):
    input_layer = Input(shape=(params['dims']))
    if(hyper_params is None):
        hyper_params=default_hyperparams()
    nn=nn_builder(params=params,
                    hyper_params=hyper_params,
                    input_layer=input_layer,
                    i=0,
                    n_cats=params['n_cats'])
    
#    loss=WeightedLoss()(specific=None,
#                       class_dict=class_dict)
    model= Model(inputs=input_layer, 
                 outputs=nn)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
    return model

def nn_builder(params,
               hyper_params,
               input_layer=None,
               i=0,
               n_cats=None):
    if(input_layer is None):
        input_layer = Input(shape=(params['dims']))
    if(n_cats is None):
        n_cats=params['n_cats']
    x_i=input_layer
    for j in range(hyper_params['layers']):
        hidden_j=int(params['dims'][0]* hyper_params[f'units_{j}'])
        x_i=Dense(hidden_j,activation='relu',
                    name=f"layer_{i}_{j}")(x_i)
    if(hyper_params['batch']):
        x_i=BatchNormalization(name=f'batch_{i}')(x_i)
    x_i=Dense(n_cats, activation='softmax',name=f'out_{i}')(x_i)
    return x_i

def default_hyperparams():
    return {'layers':1, 'units_0':2,
            'units_1':1,'batch':False}