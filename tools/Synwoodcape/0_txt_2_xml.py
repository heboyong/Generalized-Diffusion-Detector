import os
from PIL import Image
import shutil
import tqdm

root_dir = "/home/kemove/disk/data/det/SynWoodScapes/"
annotations_dir = root_dir + "box_2d_annotations/"
image_dir = root_dir + "rgb_images/"
xml_dir = root_dir + "xmls/"
# classes = ["vehicles", "person", "bicycle", "traffic_light", "traffic_sign"]
label_list = []
for filename in tqdm.tqdm(sorted(os.listdir(annotations_dir))):
    fin = open(annotations_dir + filename, 'r')
    image_name = filename.split('.')[0]
    image_path = image_dir + image_name + ".png"
    img = Image.open(image_dir + image_name + ".png")  # 若图像数据是“png”转换成“.png”即可
    width, height = img.size[0], img.size[1]
    xml_name = xml_dir + image_name + '.xml'

    lines_all = fin.readlines()
    lines = []
    number_ship = 0

    for line_all in lines_all:
        line_all = line_all.split(',')

        lines.append(line_all)

    with open(xml_name, 'w') as fout:
        fout.write('<annotation>' + '\n')
        fout.write('\t' + '<folder>VOC2007</folder>' + '\n')
        fout.write('\t' + '<filename>' + image_name + '.png' + '</filename>' + '\n')

        fout.write('\t' + '<size>' + '\n')
        fout.write('\t\t' + '<width>' + str(img.size[0]) + '</width>' + '\n')
        fout.write('\t\t' + '<height>' + str(img.size[1]) + '</height>' + '\n')
        fout.write('\t\t' + '<depth>' + '3' + '</depth>' + '\n')
        fout.write('\t' + '</size>' + '\n')

        fout.write('\t' + '<segmented>' + '0' + '</segmented>' + '\n')

        for line in lines:
            bbox = line
            class_name = str(bbox[0])
            # assert classes[class_id] == class_name

            x1 = eval(bbox[2])
            y1 = eval(bbox[3])
            x2 = eval(bbox[4])
            y2 = eval(bbox[5].strip('\n'))

            fout.write('\t' + '<object>' + '\n')
            fout.write('\t\t' + '<name>' + class_name + '</name>' + '\n')
            fout.write('\t\t' + '<pose>' + 'Unspecified' + '</pose>' + '\n')
            fout.write('\t\t' + '<truncated>' + '0' + '</truncated>' + '\n')
            fout.write('\t\t' + '<difficult>' + '0' + '</difficult>' + '\n')
            fout.write('\t\t' + '<bndbox>' + '\n')
            fout.write('\t\t\t' + '<xmin>' + str(x1) + '</xmin>' + '\n')
            fout.write('\t\t\t' + '<ymin>' + str(y1) + '</ymin>' + '\n')
            # pay attention to this point!(0-based)
            fout.write('\t\t\t' + '<xmax>' + str(x2) + '</xmax>' + '\n')
            fout.write('\t\t\t' + '<ymax>' + str(y2) + '</ymax>' + '\n')
            fout.write('\t\t' + '</bndbox>' + '\n')
            fout.write('\t' + '</object>' + '\n')

        fin.close()
        fout.write('</annotation>')


