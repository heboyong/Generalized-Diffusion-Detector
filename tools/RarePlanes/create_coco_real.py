import argparse
import pandas as pd

from solaris.data.coco import geojson2coco


def create_coco_annotations(image_dir, geojson_dir, output_path, category_attribute, custom_class_lookup_csv):
    """ parse the geojson files and create the coco annotations file
    Args:
        - image_dir (str): directory containing the tiled images
        - geojson_dir (str): directory containing the geojson files
        - output_path (str): json file containing the coco annotations
        - category_attribute (str): One of ['role','num_engines', 'propulsion', 'canards', 'num_tail_fins',
       'wing_position', 'wing_type', 'faa_wingspan_class', 'custom_id'].  Note that 'custom_id' must be first
       created using the create_custom_classes function.
        - custom_class_lookup_csv(str): path to the output csv from the create_custom_classes
        function.  Must be included if using the custom_id to create coco labels.
    Returns
        - annotations (list): list of coco annotations
    """

    if category_attribute is None:
        preset_categories = [{'id': 0, 'name': 'aircraft'}]
    else:
        if category_attribute == "num_engines":
            preset_categories = [{'id': 0, 'name': 0}, {'id': 1, 'name': 1},
                                 {'id': 2, 'name': 2}, {'id': 3, 'name': 3},
                                 {'id': 4, 'name': 4}]
        elif category_attribute == "role_id":
            preset_categories = [{'id': 0, 'name': 1}, {'id': 1, 'name': 2},
                                 {'id': 2, 'name': 3}, {'id': 3, 'name': 4},
                                 {'id': 4, 'name': 5}, {'id': 5, 'name': 6},
                                 {'id': 6, 'name': 7}]
        elif category_attribute == "role":
            print("switiching category attribute from role to role_id for easy indexing")
            category_attribute = "role_id"
            preset_categories = [{'id': 0, 'name': 1}, {'id': 1, 'name': 2},
                                 {'id': 2, 'name': 3}, {'id': 3, 'name': 4},
                                 {'id': 4, 'name': 5}, {'id': 5, 'name': 6},
                                 {'id': 6, 'name': 7}]
        elif category_attribute == "canards":
            preset_categories = [{'id': 0, 'name': 'no'}, {'id': 1, 'name': 'yes'}]
        elif category_attribute == "propulsion":
            preset_categories = [{'id': 0, 'name': 'jet'}, {'id': 1, 'name': 'propeller'},
                                 {'id': 2, 'name': 'unpowered'}]
        elif category_attribute == "num_tail_fins":
            preset_categories = [{'id': 0, 'name': 0}, {'id': 1, 'name': 1},
                                 {'id': 2, 'name': 2}, {'id': 3, 'name': 3},
                                 {'id': 4, 'name': 4}]
        elif category_attribute == "wing_position":
            preset_categories = [{'id': 0, 'name': 'delta'}, {'id': 1, 'name': 'straight'},
                                 {'id': 2, 'name': 'swept'}, {'id': 3, 'name': 'variable swept'}]
        elif category_attribute == "faa_wingspan_class":
            preset_categories = [{'id': 0, 'name': 1}, {'id': 1, 'name': 2},
                                 {'id': 2, 'name': 3}, {'id': 3, 'name': 4},
                                 {'id': 4, 'name': 5}, {'id': 5, 'name': 6}]
        elif category_attribute == "custom_id":
            if custom_class_lookup_csv is None:
                print("Please read in custom_class_lookup.csv output from the create_custom_classes function")
            else:
                lookup_csv = pd.read_csv(custom_class_lookup_csv)
                custom_list = list(lookup_csv['custom_id'])
                custom_list.sort()
                preset_categories = []
                id = 0
                for integer in custom_list:
                    preset_categories.append({'id': id, 'name': integer})
                    id += 1
        else:
            print("Unknown category specified please use one of the following:")
            print("num_engines, role_id, role, canards, propulsion, num_tail_fins,")
            print("wing_position, wing_type, faa_wingspan_class, or custom_id")
            exit()
    _ = geojson2coco(image_dir,
                     geojson_dir,
                     output_path=output_path,
                     image_ext='.png',
                     category_attribute=category_attribute,
                     preset_categories=preset_categories,
                     verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir',
                        default='/media/kemove/c74ec6f5-1534-cc40-9684-cdae3e189cf6/domain/det/RarePlanes/real/RarePlanes_test_PS-RGB_tiled/PS-RGB_tiled/')
    parser.add_argument('--geojson_dir',
                        default='/media/kemove/c74ec6f5-1534-cc40-9684-cdae3e189cf6/domain/det/RarePlanes/real/RarePlanes_test_geojson_aircraft_tiled/geojson_aircraft_tiled/')
    parser.add_argument('--output_path', default='instances_real_test.json')
    parser.add_argument('--category_attribute', default=None)
    parser.add_argument('--custom_class_lookup_csv', default=None)
    args = parser.parse_args()
    create_coco_annotations(args.image_dir,
                            args.geojson_dir,
                            args.output_path,
                            args.category_attribute,
                            args.custom_class_lookup_csv)
