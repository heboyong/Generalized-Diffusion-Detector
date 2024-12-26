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
    classes = ["Top Casing", "Gear", "Bottom Casing", "Carrier", "Cover", "Housing", "Planet", "Sun", "Button", "Cam",
               "Case End", "Case Middle"]
    categories = [{"name": "Top Casing", "id": 1, "supercategory": "none"},
                  {"name": "Gear", "id": 2, "supercategory": "none"},
                  {"name": "Bottom Casing", "id": 3, "supercategory": "none"},
                  {"name": "Carrier", "id": 4, "supercategory": "none"},
                  {"name": "Cover", "id": 5, "supercategory": "none"},
                  {"name": "Housing", "id": 6, "supercategory": "none"},
                  {"name": "Planet", "id": 7, "supercategory": "none"},
                  {"name": "Sun", "id": 8, "supercategory": "none"},
                  {"name": "Button", "id": 9, "supercategory": "none"},
                  {"name": "Cam", "id": 10, "supercategory": "none"},
                  {"name": "Case End", "id": 11, "supercategory": "none"},
                  {"name": "Case Middle", "id": 12, "supercategory": "none"}]
    # load ground-truth from xml annotations
    id_number = 0
    for image_id, label_file_name in enumerate(os.listdir(LABEL_PATH)):

        label_file = LABEL_PATH + label_file_name
        image_file = label_file_name.split('.')[0] + '.png'
        print(image_file)
        tree = ET.parse(label_file)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)


        images.append({
            "file_name": image_file,
            "height": height,
            "width": width,
            "id": str(image_file).split('.')[0]
        })  # id of the image. referenced in the annotation "image_id"

        for anno_id, obj in enumerate(root.iter('object')):
            name = obj.find('name').text
            bbox = obj.find('bndbox')

            cls_id = class_name.index(name) + 1
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            xlen = xmax - xmin
            ylen = ymax - ymin
            annotations.append({
                "segmentation": [],
                "area": xlen * ylen,
                "iscrowd": 0,
                "image_id": str(image_file).split('.')[0],
                "bbox": [xmin, ymin, xlen, ylen],
                "category_id": cls_id,
                "id": id_number,
                "ignore": 0
            })
            # print([image_file,image_id, cls_id, xmin, ymin, xlen, ylen])
            id_number += 1

    return {"images": images, "annotations": annotations, "categories": categories}


if __name__ == '__main__':
    LABEL_PATH = '/home/hby/project/UniverseNet-adapt/data/val/xmls/'
    classes = ["Top Casing", "Gear", "Bottom Casing", "Carrier", "Cover", "Housing", "Planet", "Sun", "Button", "Cam",
               "Case End", "Case Middle"]
    label_dict = load_load_image_labels(LABEL_PATH, classes)

    with open('val.json', 'w') as json_file:
        json_file.write(json.dumps(label_dict, ensure_ascii=False))
        json_file.close()
