from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class AirplaneLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        y_t = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((y_t, 1-y))
        else:
            return y

    def inverse_transform(self, y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(y[:, 0], threshold)
        else:
            return super().inverse_transform(y, threshold)
