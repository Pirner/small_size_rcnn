import os
import cv2
import pandas as pd
import tensorflow as tf
import shutil
from tqdm import tqdm

from rcnn.bbox_utils import get_iou
from rcnn.generator import AirplaneDataGenerator


class DataLoader:
    @staticmethod
    def load_airplane_dataset(root_dir, save_dir):
        """
        load the airplane dataset
        :param root_dir: root directory to load data from
        :param save_dir: directory to store created data in
        :return: two tensorflow sequence iterators to perform training on
        """
        cv2.setUseOptimized(True)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        annotation_dir = os.path.join(root_dir, 'annotations')
        im_dir = os.path.join(root_dir, 'images')

        annotations = sorted(os.listdir(annotation_dir))
        images = sorted(os.listdir(im_dir))

        t_image_paths = []
        labels = []

        assert len(annotations) == len(images)

        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        os.makedirs(save_dir, exist_ok=True)
        n = 0

        # iterate over the annotations to create training data
        for i, (im_entry, ann_entry) in tqdm(enumerate(zip(images, annotations))):
            im_path = os.path.join(im_dir, im_entry)
            print('Current image file treating:', im_path)
            ann_path = os.path.join(annotation_dir, ann_entry)
            if 'Lhasa' in im_path:
                continue
            df = pd.read_csv(ann_path)
            im = cv2.imread(im_path)

            gt_values = []

            for row in df.iterrows():
                x1 = int(row[1][0].split(' ')[0])
                y1 = int(row[1][0].split(' ')[1])
                x2 = int(row[1][0].split(' ')[2])
                y2 = int(row[1][0].split(' ')[3])

                gt_values.append({
                    'x1': x1,
                    'x2': x2,
                    'y1': y1,
                    'y2': y2,
                })

            ss.setBaseImage(im)
            ss.switchToSelectiveSearchFast()
            ss_results = ss.process()
            im_out = im.copy()

            true_counter = 0
            false_counter = 0

            # constants
            counter_threshold = 30

            for j, result in enumerate(ss_results):
                x, y, w, h = result

                pos_flag = False
                # print(x, y, w, h)
                ss_bbox = {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h}
                for gt_bbox in gt_values:
                    # store a maximum of 30 positive and 30 negative results per image (to avoid too many samples)
                    iou = get_iou(gt_bbox, ss_bbox)

                    # if the ground truth bounding box has enough shared area, then it can be saved
                    if iou > 0.70 and true_counter < counter_threshold:
                        timage = im_out[y:y + h, x:x + w]
                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)

                        t_image_path = os.path.join(save_dir, '{:06d}.png'.format(n))
                        cv2.imwrite(t_image_path, resized)
                        t_image_paths.append(t_image_path)
                        labels.append((0, 1))
                        # write label
                        label_path = os.path.join(save_dir, '{:06d}.txt'.format(n))
                        f = open(label_path, 'a')
                        f.write('1')
                        f.close()
                        n += 1
                        true_counter += 1
                        pos_flag = True
                        break

                # it the candidate is not a positive one and there is still room for negative samples, save it
                if false_counter < counter_threshold and not pos_flag:
                    timage = im_out[y:y + h, x:x + w]
                    resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                    t_image_path = os.path.join(save_dir, '{:06d}.png'.format(n))
                    cv2.imwrite(os.path.join(save_dir, '{:06d}.png'.format(n)), resized)
                    t_image_paths.append(t_image_path)
                    labels.append((1, 0))
                    # write label
                    label_path = os.path.join(save_dir, '{:06d}.txt'.format(n))
                    f = open(label_path, 'a')
                    f.write('0')
                    f.close()

                    n += 1
                    false_counter += 1

            if i > 10:
                break

        train_gen = AirplaneDataGenerator(t_image_paths, labels, batch_size=2)
        val_gen = None

        return train_gen, val_gen

    @staticmethod
    def load_data_csv_legacy(root_dir, annotation_dir):
        train_images = []
        train_labels = []
        cv2.setUseOptimized(True)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        for e, i in tqdm(enumerate(os.listdir(annotation_dir))):
            try:
                # if i.startswith('airplane'):
                filename = i.split('.')[0] + '.jpg'
                print(e, filename)
                im = cv2.imread(os.path.join(root_dir, filename))
                df = pd.read_csv(os.path.join(annotation_dir, i))

                gt_values = []

                for row in df.iterrows():
                    x1 = int(row[1][0].split(' ')[0])
                    y1 = int(row[1][0].split(' ')[1])
                    x2 = int(row[1][0].split(' ')[2])
                    y2 = int(row[1][0].split(' ')[3])

                    gt_values.append({
                        'x1': x1,
                        'x2': x2,
                        'y1': y1,
                        'y2': y2,
                    })

                ss.setBaseImage(im)
                ss.switchToSelectiveSearchFast()
                ss_results = ss.process()
                im_out = im.copy()
                counter = 0
                false_counter = 0
                flag = 0
                lflag = 0
                bitflag = 0

                for j, result in enumerate(ss_results):
                    try:
                        if j < 2000 and flag == 0:
                            for gt_val in gt_values:
                                x, y, w, h = result
                                iou = get_iou(gt_val, {'x1': x, 'x2': x + w, 'y1': y, 'y2': y + h})
                                if counter < 30:
                                    if iou > 0.70:
                                        timage = im_out[y:y + h, x:x + w]
                                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                        train_images.append(resized)
                                        train_labels.append(1)
                                        counter += 1
                                else:
                                    bitflag = 1
                                if false_counter < 30:
                                    if iou < 0.3:
                                        timage = im_out[y:y + h, x:x + w]
                                        resized = cv2.resize(timage, (224, 224), interpolation=cv2.INTER_AREA)
                                        train_images.append(resized)
                                        train_labels.append(0)
                                        false_counter += 1
                                else:
                                    bitflag = 1

                            if lflag == 1 and bitflag == 1:
                                print('inside')
                                flag = 1
                    except Exception as e:
                        print('error in filename: {0} with error {1}'.format(filename, str(e)))
                        continue

            except Exception as e:
                print('error in filename: {0} with error {1}'.format(filename, str(e)))
                continue

            return train_images, train_labels
