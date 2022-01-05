import tensorflow as tf
import numpy as np


class AirplaneDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, im_paths, labels, batch_size=1, shuffle=True):
        self._shuffle = shuffle
        self._batch_size = batch_size
        if len(labels) != len(im_paths):
            raise Exception('not equal length of labels and image paths provided.')
        self._labels = labels
        self._im_paths = im_paths
        self._data = list(zip(im_paths, labels))
        self._indexes = None

    def on_epoch_end(self):
        if self._shuffle:
            self._indexes = np.arange(len(self._data) / self._batch_size)

    def __get_data(self, indexes):
        pass

    def __getitem__(self, index):
        im_paths = self._im_paths[index * self._batch_size:(index + 1) * self._batch_size]
        labels = self._labels[index * self._batch_size:(index + 1) * self._batch_size]

    def __len__(self):
        return len(self._im_paths) // self._batch_size
