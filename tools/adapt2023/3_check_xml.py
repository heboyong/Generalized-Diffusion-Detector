import os
import os.path
import numpy as np
import xml.etree.ElementTree as xmlET
from PIL import Image, ImageDraw
import cv2

'''
[{"name": "Top Casing", "id": 1, "supercategory": "Helical Gear"}, 
 {"name": "Gear", "id": 2, "supercategory": "Helical Gear"}, 
 {"name": "Bottom Casing", "id": 3, "supercategory": "Helical Gear"}, 
 {"name": "Carrier", "id": 4, "supercategory": "Planetary Reducer"}, 
 {"name": "Cover", "id": 5, "supercategory": "Planetary Reducer"}, 
 {"name": "Housing", "id": 6, "supercategory": "Planetary Reducer"}, 
 {"name": "Planet", "id": 7, "supercategory": "Planetary Reducer"}, 
 {"name": "Sun", "id": 8, "supercategory": "Planetary Reducer"}, 
 {"name": "Button", "id": 9, "supercategory": "Toggly Fidget Button"}, 
 {"name": "Cam", "id": 10, "supercategory": "Toggly Fidget Button"}, 
 {"name": "Case End", "id": 11, "supercategory": "Toggly Fidget Button"}, 
 {"name": "Case Middle", "id": 12, "supercategory": "Toggly Fidget Button"}]
'''

# 把下面的路径改为自己的路径即可
file_path_img = '/home/hby/project/UniverseNet-adapt/data/val/images/'
file_path_xml = '/home/hby/project/UniverseNet-adapt/data/val/xmls/'
image_list = os.listdir(file_path_img)
save_file_path = 'output'

pathDir = os.listdir(file_path_xml)
for idx in range(len(pathDir)):
    filename = pathDir[idx]
    print(filename)
    tree = xmlET.parse(os.path.join(file_path_xml, filename))
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = []

    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = int(bbox.find('xmin').text)
        y1 = int(bbox.find('ymin').text)
        x2 = int(bbox.find('xmax').text)
        y2 = int(bbox.find('ymax').text)

        cla = obj.find('name').text

        box = [x1, y1, x2, y2, cla]

        boxes.append(box)
    print(box)
    image_name = os.path.splitext(filename)[0]
    for image_ in image_list:
        if image_name in image_:
            image_file = image_

    image_path = os.path.join(file_path_img, image_file)
    image = cv2.imread(image_path)

    for box in boxes:
        xmin, ymin, xmax, ymax, label = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=1)
    out_path = os.path.join(save_file_path, image_name + '.jpg')
    cv2.imwrite(out_path, image)
