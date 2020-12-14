# -*- coding: utf-8 -*-
"""
Some class extensions from keras needed for run_ann.py

Created on Mon Dec 14 10:32:51 2020

@author: lbechberger
"""

import tensorflow as tf
import time

# based on https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras
class SaltAndPepper(tf.keras.layers.Layer):
    
    def __init__(self, ratio, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio

    def call(self, inputs, training=None):
        def noised():
            shp = tf.keras.backend.shape(inputs)[1:]
            mask_select = tf.keras.backend.random_binomial(shape=shp, p=self.ratio)
            mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.5) # salt and pepper have the same chance
            out = inputs * (1-mask_select) + mask_noise * mask_select
            return out
    
        return tf.keras.backend.in_train_phase(noised(), inputs, training=training)
    
    def get_config(self):
        config = {'ratio': self.ratio}
        base_config = super(SaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        

# based on grid field guide by Julius Schöning
class AutoRestart(tf.keras.callbacks.Callback):
    '''Restart the script if walltime might be reached in the next epoch
    '''
    def __init__(self, filepath, start_time, verbose=0, walltime=0):
        super(AutoRestart, self).__init__()

        self.filepath = filepath
        self.start_time = start_time
        self.walltime = walltime
        self.wait = 0
        self.verbose = verbose
        self.epoch_start = 0
        self.epoch_average = 0
        self.stopped_epoch = 1
        self.reachedWalltime = False

    def on_train_begin(self, logs={}):
        if self.verbose > 0:
            print('qsubRestart Callback')

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start=time.time()

    def on_epoch_end(self, epoch, logs={}):
        epochtime = (time.time() - self.epoch_start)
        if self.epoch_average > 0:
            self.epoch_average = (self.epoch_average + epochtime) / 2
        else:
            self.epoch_average = epochtime
        if (time.time() - self.start_time + 3*self.epoch_average)>self.walltime:
            print("will run over walltime: %s s" % int(time.time() - self.start_time + 3*self.epoch_average))
            self.reachedWalltime = True
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.model.save_weights(self.filepath+str(epoch)+'.hdf5', overwrite=True)
        #else:
            #print("walltime in: %s s" % int(self.walltime - (time.time() - self.start_time)))

        if self.verbose > 0:
            print("-Runtime: %s s, Epoch runtime %s s, Average Epoch runtime %s s ---" % (int((time.time() - self.start_time)), int(epochtime) , int(self.epoch_average)) )


    def on_train_end(self, logs={}):
        epochtime = (time.time() - self.epoch_start)
        if self.verbose > 0:
            print("-Total Runtime: %s s, Epoch runtime %s s, Average Epoch runtime %s s---" % (int((time.time() - self.start_time)), int(epochtime) , int(self.epoch_average)) )