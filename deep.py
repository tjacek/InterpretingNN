import numpy as np
import tensorflow as tf
import keras
from keras.layers import Concatenate,Dense,BatchNormalization
from keras import Input, Model
#from tabpfn import TabPFNClassifier
import clf,dataset


class MLP(clf.Clf):
    NAME="MLP"
    def __init__( self,
                  hyper_params=None,
                  model=None):
        if(hyper_params is None):
            hyper_params={'layers':1, 'units_0':2,
                          'units_1':1,'batch':False}
        self.hyper_params=hyper_params
        self.model=model
        self.data_params=None

    def fit(self,X,y):
        self.data_params=dataset.DatasetParams.from_arr(X,y)
        self.model= self._build()
        y=tf.one_hot(y,depth=self.data_params.cats)
        return self.model.fit(x=X,
                              y=y,
                              class_weight=None,#self.class_weight,
                              epochs=1000,
                              callbacks=self.callback(),
                              verbose=False)
    def predict(self,X):
        y=self.model.predict(X,
                             verbose=False)
        return np.argmax(y,axis=1)
    
    def predict_proba(self,X):
        return self.model.predict(X,
                             verbose=False)

    def save(self,out_path):
        self.model.save(f"{out_path}.keras")
    
    @classmethod
    def callback(cls):
        return tf.keras.callbacks.EarlyStopping(monitor='accuracy', 
                                            patience=15)

    def _build( self):
        input_layer = Input(shape=(self.data_params.feats,))
        x_i=input_layer
        for i in range(self.hyper_params['layers']):
            hidden_i=int( self.data_params.feats* 
                          self.hyper_params[f'units_{i}'])
            x_i=Dense(hidden_i,activation='relu',
                    name=f"layer_{i}")(x_i)
        if(self.hyper_params['batch']):
            x_j=BatchNormalization(name=f'batch')(x_i)
        output_layer=Dense(self.data_params.cats, 
                           activation='softmax',
                           name=f'out')(x_i)
        model= Model(inputs=input_layer, 
                     outputs=output_layer)
        model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'],
                  jit_compile=False)
        return model

    @classmethod
    def read(cls,in_path):
        model=keras.saving.load_model(in_path)
        return cls(model=model)

    def get_weights(self,layer_index=1):
        layer=self.model.layers[layer_index]
        return layer.get_weights()[0]

    def random_output( self,
                       data,
                       n_samples=1000):
        value_range=data.range()
        samples=[np.random.uniform(v_min,
                           v_max,
                           size=n_samples) 
                    for v_min,v_max in value_range]
        samples=np.array(samples).T
        return self.predict(samples)

class TabPFN(clf.Clf):
    NAME="TabPFN"
    def __init__( self):
        self.model=TabPFNClassifier()