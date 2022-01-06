import cv2
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
        data = list(self._data[indexes * self._batch_size:(indexes + 1) * self._batch_size])

        images = []
        labels = []

        for im_path, label in data:
            im = cv2.imread(im_path)
            images.append(im)
            labels.append(label)

        x_batch = np.array(images)
        y_batch = np.array(labels)
        y_batch = np.reshape(y_batch, (self._batch_size, 2))

        return x_batch, y_batch

    def __getitem__(self, indexes):
        im_paths = self._im_paths[indexes * self._batch_size:(indexes + 1) * self._batch_size]
        labels = self._labels[indexes * self._batch_size:(indexes + 1) * self._batch_size]

        return self.__get_data(indexes=indexes)

    def __len__(self):
        return len(self._data) // self._batch_size
