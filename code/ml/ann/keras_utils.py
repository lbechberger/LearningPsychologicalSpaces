# -*- coding: utf-8 -*-
"""
Some class extensions from keras needed for run_ann.py

Created on Mon Dec 14 10:32:51 2020

@author: lbechberger
"""

import tensorflow as tf
import time, pickle
import cv2
import numpy as np
import csv, os

# based on https://stackoverflow.com/questions/55653940/how-do-i-implement-salt-pepper-layer-in-keras
class SaltAndPepper(tf.keras.layers.Layer):
    
    def __init__(self, ratio, only_train = False, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.supports_masking = True
        self.ratio = ratio
        self.only_train = only_train

    def call(self, inputs, training=None):
        def noised():
            shp = tf.keras.backend.shape(inputs)[1:]
            mask_select = tf.keras.backend.random_binomial(shape=shp, p=self.ratio)
            mask_noise = tf.keras.backend.random_binomial(shape=shp, p=0.5) # salt and pepper have the same chance
            out = inputs * (1-mask_select) + mask_noise * mask_select
            return out
        
        if self.only_train:
            return tf.keras.backend.in_train_phase(noised(), inputs, training=training)
        else:
            return noised()
        
    def get_config(self):
        config = {'ratio': self.ratio, 'only_train': self.only_train}
        base_config = super(SaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        


class EarlyStoppingRestart(tf.keras.callbacks.EarlyStopping):
    '''Initialize EarlyStopping with output of CSVLogger.
    '''
    def __init__(self, monitor = 'val_loss', min_delta = 0, patience = 0, verbose = 0, mode = 'auto', logpath = None, initial_epoch = 0, modelpath = None):
        super(EarlyStoppingRestart, self).__init__(monitor = monitor, min_delta = min_delta, patience = patience, verbose = verbose, mode = mode)
        self.logpath = logpath
        self.initial_epoch = initial_epoch
        self.modelpath = modelpath

    def on_train_begin(self, logs=None):
        super(EarlyStoppingRestart, self).on_train_begin(logs)
        
        if self.logpath is not None and os.path.exists(self.logpath):
            old_best = self.best
            old_best_epoch = 0
            counter = 0
            # find best value from history, compute wait as difference
            with open(self.logpath, 'r') as f_in:
                reader = csv.DictReader(f_in, delimiter=',')
                     
                for row in reader:
                    if row[self.monitor] is not None:
                        value = float(row[self.monitor])
                        epoch = int(row['epoch'])
                        # ignore all epochs later than the initial epoch - probably from failed earlier run
                        if epoch < self.initial_epoch and self.monitor_op(value, old_best):
                            counter += 1
                            old_best = value
                            old_best_epoch = epoch

            if counter > 0:
                self.wait = self.initial_epoch - old_best_epoch - 1
                self.best = old_best
                self.best_epoch = old_best_epoch
                if self.verbose > 0:
                    print('Loaded best value from CSV: {0} (wait: {1})'.format(self.best, self.wait))
 
            if self.wait < 0:
                raise Exception('Inconsistent epoch information!')

    def on_epoch_end(self, epoch, logs=None):
        super(EarlyStoppingRestart, self).on_epoch_end(epoch, logs)

        # store currently best epoch
        if self.wait == 0:
            self.best_epoch = epoch


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
#            self.model.save_weights(self.filepath+str(epoch)+'.hdf5', overwrite=True)
        #else:
            #print("walltime in: %s s" % int(self.walltime - (time.time() - self.start_time)))

        if self.verbose > 0:
            print("-Runtime: %s s, Epoch runtime %s s, Average Epoch runtime %s s ---" % (int((time.time() - self.start_time)), int(epochtime) , int(self.epoch_average)) )

        with open(self.filepath + str(epoch) + '.hdf5.opt', 'wb') as f_out:
            pickle.dump(tf.keras.backend.batch_get_value(getattr(self.model.optimizer, 'weights')), f_out)


    def on_train_end(self, logs={}):
        epochtime = (time.time() - self.epoch_start)
        if self.verbose > 0:
            print("-Total Runtime: %s s, Epoch runtime %s s, Average Epoch runtime %s s---" % (int((time.time() - self.start_time)), int(epochtime) , int(self.epoch_average)) )



class IndividualSequence(tf.keras.utils.Sequence):
    
    def __init__(self, source, info_mappers, batch_size, image_size, shuffle = True, truncate = True, mapping_function = lambda x: x):
        self._source = source
        self._info_mappers = info_mappers
        self._batch_size = batch_size
        self._image_size = image_size
        self._shuffle = shuffle
        self._truncate = truncate
        self._mapping_function = mapping_function
        if self._shuffle:
            self.on_epoch_end()
        
    def __len__(self):
        if self._truncate:
            return int(np.floor(self._source.shape[0] / self._batch_size))
        else:
            return int(np.ceil(self._source.shape[0] / self._batch_size))
    
    def __load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img / 255
        return self._mapping_function(img)
    
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
    
    def __init__(self, source_sequences, weights, dims, all_classes, berlin_classes, sketchy_classes, do_classification, do_mapping, do_reconstruction, truncate = True):
        self._source_sequences = source_sequences
        self._weights = weights
        self._dims = dims
        self._all_classes = all_classes
        self._berlin_classes = berlin_classes
        self._sketchy_classes = sketchy_classes
        self._do_classification = do_classification
        self._do_mapping = do_mapping
        self._do_reconstruction = do_reconstruction
        self._truncate = truncate
    
    def __len__(self):
        if self._truncate:
            return min([len(seq) for seq in self._source_sequences])
        else:
            return max([len(seq) for seq in self._source_sequences])
        
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
                seq_coords = seq_y[0] if w['mapping'] == 1 and seq_length > 0 else np.zeros((seq_length, self._dims))
                coords.append(seq_coords)
                weights_mapping.append(np.full((seq_length), fill_value = w['mapping']))
            
            if self._do_classification:
                seq_classes = seq_y[0] if w['classification'] == 1 and seq_length > 0 else np.zeros((seq_length, self._all_classes))
                all_classes.append(seq_classes)
                seq_berlin = seq_y[1] if w['berlin'] == 1 and seq_length > 0 else np.zeros((seq_length, self._berlin_classes))
                berlin_classes.append(seq_berlin)
                seq_sketchy = seq_y[1] if w['sketchy'] == 1 and seq_length > 0 else np.zeros((seq_length, self._sketchy_classes))
                sketchy_classes.append(seq_sketchy)
    
                weights_classification.append(np.full((seq_length), fill_value = w['classification']))
                weights_berlin.append(np.full((seq_length), fill_value = w['berlin']))
                weights_sketchy.append(np.full((seq_length), fill_value = w['sketchy']))
            
            if self._do_reconstruction:
                weights_reconstruction.append(np.full((seq_length), fill_value = w['reconstruction']))
        
        X = np.concatenate(X)
        targets = {}
        weights = {}

        # helper function to take care of fake targets in case of all zero weights
        # (relevant in last batches when one of the providers runs empty and the loss
        # for the corresponding target cannot be computed and results in NaN)
        def concat_targets_weights(label, class_targets, class_weights):
            t = np.concatenate(class_targets)
            w = np.concatenate(class_weights)
            if np.count_nonzero(t) == 0:
                t[:,0] = 1
                w[0] = 1e-10
            targets[label] = t
            weights[label] = w
        
        if self._do_mapping:
            concat_targets_weights('mapping', coords, weights_mapping)
        if self._do_classification:
            concat_targets_weights('classification', all_classes, weights_classification)
            concat_targets_weights('berlin', berlin_classes, weights_berlin)
            concat_targets_weights('sketchy', sketchy_classes, weights_sketchy)
        if self._do_reconstruction:
            concat_targets_weights('reconstruction', [X], weights_reconstruction)

        return (X, targets, weights)
        
    
    def on_epoch_end(self):
        for seq in self._source_sequences:
            seq.on_epoch_end()