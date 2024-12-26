import json
import xml.etree.ElementTree as ET
import os
import numpy as np


def load_load_image_labels(LABEL_PATH, class_name=[]):
    # temp=[]
    images = []
    type = "instances"
    annotations = []
    # assign your categories which contain the classname and calss id
    # the order must be same as the class_nmae
    categories = [

            {
                "supercategory": "vehicles",
                "id": 0,
                "name": "vehicles"
            },
            {
                "supercategory": "person",
                "id": 1,
                "name": "person"
            }
    ]

    # load ground-truth from xml annotations
    id_number = 0
    for image_id, label_file_name in enumerate(os.listdir(LABEL_PATH)):
        print(str(image_id) + ' ' + label_file_name)
        label_file = LABEL_PATH + label_file_name
        image_file = label_file_name.split('.')[0] + '.png'
        tree = ET.parse(label_file)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        images.append({
            "file_name": image_file,
            "height": height,
            "width": width,
            "id": image_id
        })  # id of the image. referenced in the annotation "image_id"

        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text

            if name == 'four-wheeler vehicle':
                name = 'vehicles'

            if name == 'pedestrian':
                name = 'person'

            if name not in class_name:
                continue

            bbox = obj.find('bndbox')
            # poly = eval(obj.find('poly').text)
            # poly.append(poly[0])
            # poly.append(poly[1])
            # print(len(poly))
            cls_id = class_name.index(name)
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xlen = xmax - xmin
            ylen = ymax - ymin
            annotations.append({
                "segmentation": [[xmin, ymin, xmin, ymax, xmax, ymax, xmax, ymin], ],
                "area": xlen * ylen,
                "iscrowd": 0,
                "image_id": image_id,
                "bbox": [xmin, ymin, xlen, ylen],
                "category_id": cls_id,
                "id": id_number,
                "ignore": 0
            })
            # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
            id_number += 1

    return {"images": images, "annotations": annotations, "categories": categories}


if __name__ == '__main__':

    classes = ["vehicles", "person"]

    LABEL_PATH = '/home/kemove/disk/data/det/SynWoodScapes/xmls/'
    label_dict = load_load_image_labels(LABEL_PATH, classes)
    with open('train_2classes.json', 'w') as json_file:
        json_file.write(json.dumps(label_dict, ensure_ascii=False))
        json_file.close()


