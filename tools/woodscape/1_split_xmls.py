import json
import os
import shutil

xml_list = sorted(os.listdir('/home/kemove/disk/data/det/WoodScapes/xmls'))

for xml_name in xml_list[:int(len(xml_list)*0.8)]:
    shutil.copy('/home/kemove/disk/data/det/WoodScapes/xmls/'+xml_name,'/home/kemove/disk/data/det/WoodScapes/train_xmls/'+xml_name)

for xml_name in xml_list[int(len(xml_list)*0.8):]:
    shutil.copy('/home/kemove/disk/data/det/WoodScapes/xmls/'+xml_name,'/home/kemove/disk/data/det/WoodScapes/test_xmls/'+xml_name)