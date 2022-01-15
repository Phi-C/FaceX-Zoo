import os
import json
import glob
import cv2
import numpy as np


def seg_result_from_labelme(src_root, dst_root, ext='png'):
    '''
    For semantic segmentation, get label images from labelme's json file.
    This function is useful when prepare training data.
    Args:
        src_root: directory that contains original images and json files
        dst_root: directory that contains label images
        ext: extension for label image
    '''
    # Determine the label for each category
    label2color = {'background': 0, 'mask': 1, 'ignore': 2}

    json_files = glob.glob(src_root + '/*.json')
    for file in json_files:
        with open(file, 'r') as f:
            content = json.load(f)
        h = content['imageHeight']
        w = content['imageWidth']
        img_name = content['imagePath']
        dst_img_path = os.path.join(dst_root, '.'.join(img_name.split('.')[:-1]) + '.{}'.format(ext))

        # Determine the number of channels for label image 
        dst_img = np.zeors((h, w, 1), dtype=np.uint8)
        shapes = content['shapes']
        for shape in shapes:
            label_name = shape['label']
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(dst_img, pts=[points], color=label2color[label_name])

        cv2.imwrite(dst_img_path, dst_img)



if __name__ == "__main__":
    src_root = ''