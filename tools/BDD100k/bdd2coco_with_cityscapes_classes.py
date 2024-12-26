import os
import json
import argparse
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description='BDD100K to COCO format')
    parser.add_argument(
        "-l", "--label_dir",
        default="/home/kemove/disk/data/det/BDD100K/bdd100k/bdd100k_labels_release/bdd100k/labels/",
        help="root directory of BDD label Json files",
    )
    parser.add_argument(
        "-s", "--save_path",
        default="",
        help="path to save coco formatted label file",
    )
    return parser.parse_args()


def bdd2coco_detection(id_dict, labeled_images, fn):
    images = list()
    annotations = list()

    counter = 0
    for i in tqdm(labeled_images):
        counter += 1
        image = dict()
        image['file_name'] = i['name']
        image['height'] = 720
        image['width'] = 1280

        image['id'] = counter

        empty_image = True

        for label in i['labels']:
            annotation = dict()


            if label['category'] == 'bike':
                label['category'] = 'bicycle'

            if label['category'] == 'motor':
                label['category'] = 'motorcycle'

            if label['category'] not in classes:
                continue


            empty_image = False
            annotation["iscrowd"] = 0
            annotation["image_id"] = image['id']
            x1 = int(label['box2d']['x1'])
            y1 = int(label['box2d']['y1'])
            x2 = int(label['box2d']['x2'])
            y2 = int(label['box2d']['y2'])
            annotation['bbox'] = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
            annotation['area'] = float((x2 - x1) * (y2 - y1))
            annotation['category_id'] = id_dict[label['category']]
            annotation['ignore'] = 0
            annotation['id'] = label['id']
            annotation['segmentation'] = []
            annotations.append(annotation)

        if empty_image:
            continue

        images.append(image)

    attr_dict["images"] = images
    attr_dict["annotations"] = annotations
    attr_dict["type"] = "instances"

    print('saving...')
    json_string = json.dumps(attr_dict, indent=4)
    with open(fn, "w") as file:
        file.write(json_string)


if __name__ == '__main__':
    args = parse_arguments()

    attr_dict = dict()
    classes = ('bicycle', 'bus', 'car', 'motorcycle', 'person', 'rider', 'train', 'truck')
    attr_dict["categories"] = [
        {"supercategory": "none", "id": 0, "name": "bicycle"},
        {"supercategory": "none", "id": 1, "name": "bus"},
        {"supercategory": "none", "id": 2, "name": "car"},
        {"supercategory": "none", "id": 3, "name": "motorcycle"},
        {"supercategory": "none", "id": 4, "name": "person"},
        {"supercategory": "none", "id": 5, "name": "rider"},
        {"supercategory": "none", "id": 6, "name": "train"},
        {"supercategory": "none", "id": 7, "name": "truck"},
    ]

    attr_id_dict = {i['name']: i['id'] for i in attr_dict['categories']}

    # create BDD training set detections in COCO format
    print('Loading training set...')
    with open(os.path.join(args.label_dir,
                           'bdd100k_labels_images_train.json')) as f:
        train_labels = json.load(f)
    print('Converting training set...')

    out_fn = os.path.join(args.save_path,
                          'train.json')
    bdd2coco_detection(attr_id_dict, train_labels, out_fn)

    print('Loading validation set...')
    # create BDD validation set detections in COCO format
    with open(os.path.join(args.label_dir,
                           'bdd100k_labels_images_val.json')) as f:
        val_labels = json.load(f)
    print('Converting validation set...')

    out_fn = os.path.join(args.save_path,
                          'val.json')
    bdd2coco_detection(attr_id_dict, val_labels, out_fn)
