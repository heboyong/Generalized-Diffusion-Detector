#! /usr/bin/env python3

import sys
import cv2

from scripts.helpers import *

class_id_to_str = {
    24: 'person',
    25: 'rider',
    26: 'car',
    27: 'truck',
    28: 'bus',
    31: 'train',
    32: 'motorcycle',
    33: 'bicycle'
}


# Functions ---


def get_box_2d():
    bbox_list = []
    for inst, box in bbox_data.items():
        # Info
        class_idx = metadata['instance']['class'][inst]
        class_name = class_id_to_str[class_idx]
        occluded = metadata['instance']['occluded'][inst]
        if occluded > 0.9:
            continue
        # Box
        xmin = box['xmin']
        xmax = box['xmax']
        ymin = box['ymin']
        ymax = box['ymax']

        if class_name == 'person':
            print('ok')

        bbox_list.append([int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h), class_name])
    return bbox_list


if __name__ == '__main__':

    path = '/media/kemove/c74ec6f5-1534-cc40-9684-cdae3e189cf6/domain/det/SynScapes/Synscapes'

    # Script ---

    root = os.path.abspath(path)

    # Ensure root exists
    if not os.path.exists(root):
        print('Invalid path:', root)
        sys.exit(1)

    img_dir, meta_dir = [os.path.join(root, x) for x in ['img', 'meta']]

    for index in range(25000):
        print(index)
        img_path = os.path.join(root, 'img', 'rgb-2k', '{}.png'.format(index + 1))
        meta_path = os.path.join(root, 'meta')

        metadata = read_metadata(meta_path, index + 1)[index + 1]

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        bbox_data = metadata['instance']['bbox{}'.format('2d')]
        bbox_list = get_box_2d()
        xml_name = os.path.join(root, "img", "xmls", str(index + 1) + '.xml')
        with open(xml_name, 'w') as fout:
            fout.write('<annotation>' + '\n')
            fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
            fout.write('\t' + '<filename>' + str(index + 1) + '.png' + '</filename>' + '\n')
            fout.write('\t' + '<size>' + '\n')
            fout.write('\t\t' + '<width>' + str(int(w)) + '</width>' + '\n')
            fout.write('\t\t' + '<height>' + str(int(h)) + '</height>' + '\n')
            fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
            fout.write('\t' + '</size>' + '\n')

            fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

            for box in bbox_list:
                xmin, ymin, xmax, ymax, class_name = box

                fout.write('\t' + '<object>' + '\n')
                fout.write('\t\t' + '<name>' + class_name + '</name>' + '\n')
                fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
                fout.write('\t\t' + '<truncated>' + '0' + '</truncated>' + '\n')
                fout.write('\t\t' + '<difficult>' + '0' + '</difficult>' + '\n')
                fout.write('\t\t' + '<bndbox>' + '\n')
                fout.write('\t\t\t' + '<xmin>' + str(xmin) + '</xmin>' + '\n')
                fout.write('\t\t\t' + '<ymin>' + str(ymin) + '</ymin>' + '\n')
                # pay attention to this point!(0-based)
                fout.write('\t\t\t' + '<xmax>' + str(xmax) + '</xmax>' + '\n')
                fout.write('\t\t\t' + '<ymax>' + str(ymax) + '</ymax>' + '\n')
                fout.write('\t\t' + '</bndbox>' + '\n')
                fout.write('\t' + '</object>' + '\n')

            fout.write('</annotation>')
