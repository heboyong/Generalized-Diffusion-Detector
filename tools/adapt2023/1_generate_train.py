import os
import shutil

root_dir = '//home/hby/project/UniverseNet-adapt/data/adapt-2023/adapt-2023/test/'
dst_dir = '/home/hby/project/UniverseNet-adapt/data/test/'

sets_dir = os.listdir(root_dir)

for image_set in sets_dir:
    image_dir = os.listdir(os.path.join(root_dir, image_set, 'rgb'))
    # json_file = os.path.join(root_dir, image_set, 'scene_gt_coco.json')
    # shutil.copy(json_file, os.path.join(dst_dir, 'jsons', str(image_set) + '-' + 'scene_gt_coco.json'))
    for image_name in image_dir:
        print(image_name)
        source = os.path.join(root_dir, image_set, 'rgb', image_name)
        target = os.path.join(dst_dir, 'images', str(image_set)[-3:] + '-' + image_name)
        shutil.copy(source, target)

