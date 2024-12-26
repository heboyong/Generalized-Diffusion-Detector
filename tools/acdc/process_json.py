import json

json_file = '/home/kemove/disk/project/cv_tools/instancesonly_train_gt_detection.json'

with open(json_file, 'r') as file:
    json_messgaes = json.load(file)

categories = [{"supercategory": "none", "id": 0, "name": "bicycle"},
              {"supercategory": "none", "id": 1, "name": "bus"},
              {"supercategory": "none", "id": 2, "name": "car"},
              {"supercategory": "none", "id": 3, "name": "motorcycle"},
              {"supercategory": "none", "id": 4, "name": "person"},
              {"supercategory": "none", "id": 5, "name": "rider"},
              {"supercategory": "none", "id": 6, "name": "train"},
              {"supercategory": "none", "id": 7, "name": "truck"}]

json_messgaes['categories'] = categories
'''
[{"id": 24, "name": "person", "supercategory": "human"}, 
 {"id": 25, "name": "rider", "supercategory": "human"},
 {"id": 26, "name": "car", "supercategory": "vehicle"}, 
 {"id": 27, "name": "truck", "supercategory": "vehicle"},
 {"id": 28, "name": "bus", "supercategory": "vehicle"}, 
 {"id": 31, "name": "train", "supercategory": "vehicle"},
 {"id": 32, "name": "motorcycle", "supercategory": "vehicle"},
 {"id": 33, "name": "bicycle", "supercategory": "vehicle"}]
'''
change_id_dict = {"24": 4, "25": 5, "26": 2, "27": 7, "28": 1, "31": 6, "32": 3, "33": 0}

index = 0
for anno in json_messgaes['annotations']:
    anno['category_id'] = change_id_dict[str(anno['category_id'])]


with open('train.json', 'w') as json_file:
    json.dump(json_messgaes, json_file, indent=2)
