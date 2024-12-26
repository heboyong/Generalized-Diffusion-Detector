import json
import os
import shutil

xml_list = os.listdir('/home/kemove/disk/data/det/adapt2023/real/xmls')
print(xml_list)

for xml_name in xml_list[:int(len(xml_list)*0.5)]:
    shutil.copy('/home/kemove/disk/data/det/adapt2023/real/xmls/'+xml_name,'/home/kemove/disk/data/det/adapt2023/real/train_xmls/'+xml_name)

for xml_name in xml_list[int(len(xml_list)*0.5):]:
    shutil.copy('/home/kemove/disk/data/det/adapt2023/real/xmls/'+xml_name,'/home/kemove/disk/data/det/adapt2023/real/test_xmls/'+xml_name)