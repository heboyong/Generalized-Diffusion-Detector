import json

json_file = '/home/kemove/disk/project/mmdet-domain-adaption/data/cityscapes/test.json'

with open(json_file, 'r') as file:
    json_messgaes = json.load(file)

categories = [{"supercategory": "none", "id": 0, "name": "vehicles"},
              {"supercategory": "none", "id": 1, "name": "person"}, ]

'''

'''
json_messgaes['categories'] = categories
'''
[
{"supercategory": "none", "id": 0, "name": "bicycle"}, 
{"supercategory": "none", "id": 1, "name": "bus"}, 
{"supercategory": "none", "id": 2, "name": "car"}, 
{"supercategory": "none", "id": 3, "name": "motorcycle"}, 
{"supercategory": "none", "id": 4, "name": "person"}, 
{"supercategory": "none", "id": 5, "name": "rider"}, 
{"supercategory": "none", "id": 6, "name": "train"}, 
{"supercategory": "none", "id": 7, "name": "truck"}
]
'''
change_id_dict = {"0": -1, "1": 0, "2": 0, "3": -1, "4": 1, "5": -1, "6": 0, "7": 0}

new_anno = []
index = 0
for anno in json_messgaes['annotations']:
    new_id = change_id_dict[str(anno['category_id'])]
    if new_id >= 0:
        anno['category_id'] = new_id
        anno['id'] = index
        new_anno.append(anno)
        index += 1

json_messgaes['annotations'] = new_anno
with open('test_for_wood.json', 'w') as json_file:
    json.dump(json_messgaes, json_file, indent=2)
