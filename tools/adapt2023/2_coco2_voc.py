import os
import json
import pandas as pd
from xml.dom.minidom import Document

annos = "/home/hby/project/UniverseNet-adapt/data/val/jsons"
xmldir = "/home/hby/project/UniverseNet-adapt/data/val/xmls/"

for anno in os.listdir(annos):
    anno_name = anno.split('-')[0]

    anno = os.path.join(annos, anno)

    with open(anno, 'r') as load_f:
        f = json.load(load_f)
    df_anno = pd.DataFrame(f['annotations'])
    imgs = f['images']
    cata = {}


    def createCate():
        global cata
        df_cate = f['categories']
        for item in df_cate:
            cata[item['id']] = item['name']


    def json2xml():
        global cata
        for im in imgs:
            filename = str(anno_name) + '-' + im['file_name'].split('/')[-1]
            print(filename)
            height = im['height']
            img_id = im['id']
            width = im['width']
            doc = Document()
            annotation = doc.createElement('annotation')
            doc.appendChild(annotation)
            filenamedoc = doc.createElement("filename")
            annotation.appendChild(filenamedoc)
            filename_txt = doc.createTextNode(filename)
            filenamedoc.appendChild(filename_txt)
            size = doc.createElement("size")
            annotation.appendChild(size)
            widthdoc = doc.createElement("width")
            size.appendChild(widthdoc)
            width_txt = doc.createTextNode(str(width))
            widthdoc.appendChild(width_txt)

            heightdoc = doc.createElement("height")
            size.appendChild(heightdoc)
            height_txt = doc.createTextNode(str(height))
            heightdoc.appendChild(height_txt)

            annos = df_anno[df_anno["image_id"].isin([img_id])]
            for index, row in annos.iterrows():
                bbox = row["bbox"]
                category_id = row["category_id"]
                cate_name = cata[category_id]

                object = doc.createElement('object')
                annotation.appendChild(object)

                name = doc.createElement('name')
                object.appendChild(name)
                name_txt = doc.createTextNode(cate_name)
                name.appendChild(name_txt)

                pose = doc.createElement('pose')
                object.appendChild(pose)
                pose_txt = doc.createTextNode('Unspecified')
                pose.appendChild(pose_txt)

                truncated = doc.createElement('truncated')
                object.appendChild(truncated)
                truncated_txt = doc.createTextNode('0')
                truncated.appendChild(truncated_txt)

                difficult = doc.createElement('difficult')
                object.appendChild(difficult)
                difficult_txt = doc.createTextNode('0')
                difficult.appendChild(difficult_txt)

                bndbox = doc.createElement('bndbox')
                object.appendChild(bndbox)

                xmin = doc.createElement('xmin')
                bndbox.appendChild(xmin)
                xmin_txt = doc.createTextNode(str(int(bbox[0])))
                xmin.appendChild(xmin_txt)

                ymin = doc.createElement('ymin')
                bndbox.appendChild(ymin)
                ymin_txt = doc.createTextNode(str(int(bbox[1])))
                ymin.appendChild(ymin_txt)

                xmax = doc.createElement('xmax')
                bndbox.appendChild(xmax)
                xmax_txt = doc.createTextNode(str(int(bbox[0] + bbox[2])))
                xmax.appendChild(xmax_txt)

                ymax = doc.createElement('ymax')
                bndbox.appendChild(ymax)
                ymax_txt = doc.createTextNode(str(int(bbox[1] + bbox[3])))
                ymax.appendChild(ymax_txt)

            xmlpath = os.path.join(xmldir, filename.replace('.png', '.xml'))

            f = open(xmlpath, "w")
            f.write(doc.toprettyxml(indent="  "))
            f.close()


    createCate()
    json2xml()
