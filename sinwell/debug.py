import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from rcnn.model import RCNNTrainWrapper
from rcnn.data_utils import DataLoader
from rcnn.label_utils import AirplaneLabelBinarizer


def main():
    """
    train an rcnn model from scratch
    :return:
    """
    # root_dataset_path = 'C:/private_projects/data/airplane_dataset'
    root_dataset_path = 'C:/dev/data/airplanes'

    path = os.path.join(root_dataset_path, 'images')
    annot = os.path.join(root_dataset_path, 'airplanes_annotations')

    # draw a sample file
    # for e, i in enumerate(os.listdir(annot)):
    #     filename = i.split('.')[0] + '.jpg'
    #     print(filename)
    #     im = cv2.imread(os.path.join(path, filename))
    #     df = pd.read_csv(os.path.join(annot, i))
    #     plt.imshow(im)
    #     for row in df.iterrows():
    #         x1 = int(row[1][0].split(" ")[0])
    #         y1 = int(row[1][0].split(" ")[1])
    #         x2 = int(row[1][0].split(" ")[2])
    #         y2 = int(row[1][0].split(" ")[3])
    #         cv2.rectangle(im, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     plt.figure()
    #     plt.imshow(im)
    #
    #     break
    # cv2.setUseOptimized(True)
    # ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    # im = cv2.imread(os.path.join(path, '42850.jpg'))
    # print(im.shape)
    # ss.setBaseImage(im)
    # ss.switchToSelectiveSearchFast()
    # rects = ss.process()
    # print(len(rects))
    # im_out = im.copy()
    #
    # for i, rect in enumerate(rects):
    #     x, y, w, h, = rect
    #     cv2.rectangle(im_out, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    # plt.imshow(im_out)
    # plt.show()

    rcnn_trainer = RCNNTrainWrapper()
    train_images, train_labels = DataLoader.load_data_csv_legacy(path, annot)

    x_new = np.array(train_images)
    y_new = np.array(train_labels)

    l_enc = AirplaneLabelBinarizer()
    y = l_enc.fit_transform(y_new)

    x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.1)

    tr_data = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90
    )

    train_data = tr_data.flow(x=x_train, y=y_train)
    ts_data = tf.keras.preprocessing.image.ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=90,
    )

    test_data = ts_data.flow(x=x_test, y=y_test)

    print(len(train_images))


if __name__ == '__main__':
    main()
