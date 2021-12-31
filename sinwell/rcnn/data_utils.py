import os
import cv2
import pandas as pd

from rcnn.bbox_utils import get_iou


class DataLoader:
    @staticmethod
    def load_data_csv_legacy(root_dir, annotation_dir):
        train_images = []
        train_labels = []
        cv2.setUseOptimized(True)
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

        for e, i in enumerate(os.listdir(annotation_dir)):
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
