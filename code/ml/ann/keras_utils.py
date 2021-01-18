# -*- coding: utf-8 -*-
"""
Some class extensions from keras needed for run_ann.py

Created on Mon Dec 14 10:32:51 2020

@author: lbechberger
"""

import tensorflow as tf
import time
import cv2
import numpy as np
import csv, os

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


class EarlyStoppingRestart(tf.keras.callbacks.EarlyStopping):
    '''Initialize EarlyStopping with output of CSVLogger.
    '''
    def __init__(self, monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto', filepath = None, initial_epoch = 0):
        super(EarlyStoppingRestart, self).__init__(monitor = monitor, min_delta = min_delta, patience = patience, verbose = verbose, mode = mode)
        self.filepath = filepath
        self.initial_epoch = initial_epoch

    def on_train_begin(self, logs=None):
        super(EarlyStoppingRestart, self).on_train_begin(logs)
        
        if self.filepath is not None and os.path.exists(self.filepath):
            old_best = self.best
            old_best_epoch = 0
            counter = 0
            # find best value from history, compute wait as difference
            with open(self.filepath, 'r') as f_in:
                reader = csv.DictReader(f_in, delimiter=',')
                     
                for row in reader:
                    counter += 1
                    if self.monitor_op(float(row[self.monitor]), old_best):
                        old_best = float(row[self.monitor])
                        old_best_epoch = int(row['epoch'])

            if counter > 0:
                self.wait = self.initial_epoch - old_best_epoch - 1
                self.best = old_best
                if self.verbose > 0:
                    print('Loaded best value from CSV: {0} (wait: {1})'.format(self.best, self.wait))
 
            if self.wait < 0:
                raise Exception('Inconsistent epoch information!')


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
        self.reached_walltime = False

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
            self.reached_walltime = True
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



class IndividualSequence(tf.keras.utils.Sequence):
    
    def __init__(self, source, info_mappers, batch_size, image_size, shuffle = True):
        self._source = source
        self._info_mappers = info_mappers
        self._batch_size = batch_size
        self._image_size = image_size
        self._shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(self._source.shape[0] / self._batch_size))
    
    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return img        
    
    def __getitem__(self, idx):
        current_selection = self._source[idx * self._batch_size : (idx + 1) * self._batch_size]
        X = []
        ys = []
        for info_mapper in self._info_mappers:
            ys.append([])
        
        for path, info in current_selection:
            X.append(self.__load_image(path))
            for idx, info_mapper in enumerate(self._info_mappers):
                ys[idx].append(info_mapper[info])
        X = np.reshape(X, (-1, self._image_size, self._image_size, 1))
        for i in range(len(ys)):
            ys[i] = np.array(ys[i])
        return X, ys
    
    def on_epoch_end(self):
        if self._shuffle == True:
            np.random.shuffle(self._source)
    

    
class OverallSequence(tf.keras.utils.Sequence):
    '''Sequence for dynamically loading the augmented Shapes dataset and combining its different sources.
    '''
    
    def __init__(self, source_sequences, weights, dims, all_classes, berlin_classes, sketchy_classes, do_classification, do_mapping, do_reconstruction):
        self._source_sequences = source_sequences
        self._weights = weights
        self._dims = dims
        self._all_classes = all_classes
        self._berlin_classes = berlin_classes
        self._sketchy_classes = sketchy_classes
        self._do_classification = do_classification
        self._do_mapping = do_mapping
        self._do_reconstruction = do_reconstruction
    
    def __len__(self):
        return min([len(seq) for seq in self._source_sequences])
        
    def __getitem__(self, idx):
        
        X = []
        if self._do_mapping:
            coords = []
            weights_mapping = []
        if self._do_classification:
            all_classes = []
            berlin_classes = []
            sketchy_classes = []
            weights_classification = []
            weights_berlin = []
            weights_sketchy = []
        if self._do_reconstruction:
            weights_reconstruction = []
        
        for seq, w in zip(self._source_sequences, self._weights):
            seq_X, seq_y = seq[idx]
            seq_length = seq_y[0].shape[0]
            X.append(seq_X)
            
            if self._do_mapping:
                seq_coords = seq_y[0] if w['mapping'] == 1 else np.zeros((seq_length, self._dims))
                coords.append(seq_coords)
                weights_mapping.append(np.full((seq_length), fill_value = w['mapping']))
            
            if self._do_classification:
                seq_classes = seq_y[0] if w['classification'] == 1 else np.zeros((seq_length, self._all_classes))
                all_classes.append(seq_classes)
                seq_berlin = seq_y[1] if w['berlin'] == 1 else np.zeros((seq_length, self._berlin_classes))
                berlin_classes.append(seq_berlin)
                seq_sketchy = seq_y[1] if w['sketchy'] == 1 else np.zeros((seq_length, self._sketchy_classes))
                sketchy_classes.append(seq_sketchy)
    
                weights_classification.append(np.full((seq_length), fill_value = w['classification']))
                weights_berlin.append(np.full((seq_length), fill_value = w['berlin']))
                weights_sketchy.append(np.full((seq_length), fill_value = w['sketchy']))
            
            if self._do_reconstruction:
                weights_reconstruction.append(np.full((seq_length), fill_value = w['reconstruction']))
        
        X = np.concatenate(X)
        targets = {}
        weights = {}
        
        if self._do_mapping:
            targets['mapping'] = np.concatenate(coords)
            weights['mapping'] = np.concatenate(weights_mapping)
        if self._do_classification:
            targets['classification'] = np.concatenate(all_classes)
            weights['classification'] = np.concatenate(weights_classification)
            targets['berlin'] = np.concatenate(berlin_classes)
            weights['berlin'] = np.concatenate(weights_berlin)
            targets['sketchy'] = np.concatenate(sketchy_classes)
            weights['sketchy'] = np.concatenate(weights_sketchy)
        if self._do_reconstruction:
            targets['reconstruction'] = X
            weights['reconstruction'] = np.concatenate(weights_reconstruction)

        return (X, targets, weights)
        
    
    def on_epoch_end(self):
        for seq in self._source_sequences:
            seq.on_epoch_end()