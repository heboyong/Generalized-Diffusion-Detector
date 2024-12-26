import argparse
import glob
import json
import logging
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from xmls import parse_xml_image, get_polygon


def create_coco_annotations(data_dir, segmentation, output_path, category_attribute, custom_class_lookup_csv):
    """ parse the geojson files and create the coco annotations file
    Args:
        - data_dir (str): the root directory contatining the synthetic images, masks, and annotations.
          Typically /synthetic/train/ or /synthetic/test/
          Must have 3 subdirs including: 'xmls', 'masks', and 'images'.
        - segmentation (str): Type of segmentation, choose between none, simple or full
        - output_path (str): json file containing the coco annotations
        - category_attribute (str): One of 'role','num_engines', 'propulsion', 'canards', 'num_tail_fins',
       'wing_position', 'wing_type', 'faa_wingspan_class', or 'custom_id'.  Note that 'custom_id' must be first
       created on the real dataset using the create_custom_classes function.
        - custom_class_lookup_csv(str): path to the output csv from the create_custom_classes
        function.  Must be included if using the custom_id to create coco labels.
    """
    # load the conversion table. we can get the planes attributes from their names
    conversion_table = None

    # here we decide of the mapping
    if category_attribute is None:
        categories = [{'id': 1, 'name': 'aircraft'}]
    else:
        if category_attribute == "num_engines":
            categories = {0, 1, 2, 3, 4}
        elif category_attribute == "role_id":
            categories = {1, 2, 3, 4, 5, 6, 7}
        elif category_attribute == "role":
            category_attribute = "role_id"
            categories = {1, 2, 3, 4, 5, 6, 7}
        elif category_attribute == "canards":
            categories = {"no", "yes"}
        elif category_attribute == "propulsion":
            categories = {"jet", "propeller", "unpowered"}
        elif category_attribute == "num_tail_fins":
            categories = {0, 1, 2, 3, 4}
        elif category_attribute == "wing_position":
            categories = {"high mounted", "mid/low mounted"}
        elif category_attribute == "wing_type":
            categories = {"delta", "straight", "swept", "variable swept"}
        elif category_attribute == "faa_wingspan_class":
            categories = {1, 2, 3, 4, 5, 6}
        elif category_attribute == "custom_id":
            if custom_class_lookup_csv is None:
                print("Please read in custom_class_lookup.csv output from the create_custom_classes function")
            else:
                lookup_csv = pd.read_csv(custom_class_lookup_csv)
                custom_list = list(lookup_csv['custom_id'])
                custom_list.sort()
                category_attributes = list(lookup_csv.columns)
                category_attributes.remove("custom_id")
                category_attributes.remove("Unnamed: 0")
                conversion_table = pd.merge(conversion_table, lookup_csv, on=category_attributes, how='left')
                conversion_table.sort_values(by=['custom_id'], inplace=True)
                max_val = np.max(conversion_table['custom_id'])
                fixed_ids = []
                for _, row in conversion_table.iterrows():
                    if row['custom_id'] != row['custom_id']:  # Fix NANs
                        max_val += 1
                        fixed_ids.append(max_val)
                        custom_list.append(max_val)
                    else:
                        fixed_ids.append(row['custom_id'])
                categories = set(custom_list)
                conversion_table['custom_id'] = fixed_ids
        else:
            print("Unknown category specified please use one of the following:")
            print("num_engines, role_id, role, canards, propulsion, num_tail_fins,")
            print("wing_position, wing_type, faa_wingspan_class, or custom_id")
            exit()
        categories = [{'id': i, 'name': n} for i, n in enumerate(categories)]
        cat2id = {k['name']: k['id'] for k in categories}
        make2id = {'_'.join(make.split(' ')): cat2id[cat] for make, cat in
                   zip(conversion_table['reverie_make'], conversion_table[category_attribute]) if isinstance(make, str)}

    logger = logging.getLogger(__name__)
    assert segmentation in ['none', 'simple', 'full'], 'Wrong segmentation type.'

    xml_dir = os.path.join(data_dir, "xmls")
    xml_paths = glob.glob(xml_dir + '/*.xml')
    print('Now processing', len(xml_paths), 'files')
    logger.info(f'Now processing {len(xml_paths)} files')

    ann_id = 0
    images = []
    annotations = []
    for i, path in enumerate(tqdm(xml_paths)):
        # get path of corresponding image and mask
        im_path = path.replace('.xml', '.png')
        im_path = im_path.replace('/xmls/', '/images/')
        mask_path = path.replace('.xml', '_mask.png')
        mask_path = mask_path.replace('/xmls/', '/masks/')
        if not os.path.isfile(im_path):
            logger.info(f'Missing image for {os.path.basename(path)} file')
            continue
        if not os.path.isfile(mask_path):
            logger.info(f'Missing mask for {os.path.basename(path)} file')
            continue

        # parse the xml file
        df = parse_xml_image(path)

        if df is None:
            continue

        # get the bboxes
        bboxes = np.array([df['xmin'], df['xmax'], df['ymin'], df['ymax']]).transpose()

        # get the segmentation
        colors = None
        if segmentation == 'simple':
            polygons = df['segmentation']
        elif segmentation == 'full':
            mask = cv2.imread(mask_path)[..., ::-1]
            colors = df['object_mask_color_rgba']

        # create images metadata
        width, height = 1920, 1080  # default resolution of the synthetic images
        im_name = os.path.basename(path).replace('xml', 'png')
        im_id = i + 1
        image_dic = {'width': width,
                     'height': height,
                     'file_name': im_name,
                     'date_captured': None,
                     'coco_url': None,
                     'flickr_url': None,
                     'license': None,
                     'id': im_id}
        images.append(image_dic)

        # create the annotations
        for idx, bb, in enumerate(bboxes):
            # get the plane make
            make = df['make'][idx]
            cat_id = 1 if category_attribute is None else make2id[make]

            x1, x2, y1, y2 = bb.tolist()
            ann_dic = {'bbox': [x1, y1, x2 - x1, y2 - y1],
                       'area': (x2 - x1) * (y2 - y1),
                       'category_id': cat_id,
                       'image_id': im_id,
                       'iscrowd': 0,
                       'id': ann_id
                       }
            if segmentation == 'simple':
                ann_dic['segmentation'] = [polygons[idx]]
            elif segmentation == 'full':
                color = colors[idx]
                seg = get_polygon(color, mask)
                ann_dic['segmentation'] = [seg]
            ann_id += 1
            annotations.append(ann_dic)

    coco = {}
    coco['images'] = images
    coco['annotations'] = annotations
    coco['categories'] = categories
    with open(output_path, 'w') as f:
        json.dump(coco, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        default='/media/kemove/c74ec6f5-1534-cc40-9684-cdae3e189cf6/domain/det/RarePlanes/synthetic/train')
    parser.add_argument('--segmentation', default='simple',
                        help='Type of segmentation, choose between none, simple or full')
    parser.add_argument('--output_path', default='instances_synthetic_train.json',
                        help='json file path')
    parser.add_argument('--category_attribute', default=None)
    parser.add_argument('--custom_class_lookup_csv', default=None)
    args = parser.parse_args()
    create_coco_annotations(args.data_dir,
                            args.segmentation,
                            args.output_path,
                            args.category_attribute,
                            args.custom_class_lookup_csv)
