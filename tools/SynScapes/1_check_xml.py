import os.path
import xml.etree.ElementTree as xmlET
import cv2
# 把下面的路径改为自己的路径即可

root_dir = '/media/kemove/c74ec6f5-1534-cc40-9684-cdae3e189cf6/domain/det/SynScapes/Synscapes/img/'
file_path_img = root_dir+'rgb-2k/'
file_path_xml = root_dir+'xmls/'
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


    image_path = os.path.join(file_path_img, image_name+'.png')
    image = cv2.imread(image_path)

    for box in boxes:
        xmin, ymin, xmax, ymax, label = box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness=3)
    out_path = os.path.join(save_file_path, image_name + '.jpg')
    cv2.imwrite(out_path, image)
