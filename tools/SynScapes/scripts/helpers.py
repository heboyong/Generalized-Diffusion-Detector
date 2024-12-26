import json
import math
import os

import numpy as np

# Constants ---

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)    

# Functions ---
    
def __json_to_dict(fn):
    return json.loads(''.join(open(fn).readlines()))


def stride(num, desired):
    '''Computes the stride to be used in array slicing in order to subsample a list'''
    return max(1, int(math.ceil(num / desired)))


def read_metadata(path, indices):
    '''
    Reads metadata for the specified indices and returns a dictionary
    mapping the index to the metadata dictionary.
    '''
    if isinstance(indices, int):
        indices = [indices]
    return {idx : __json_to_dict(os.path.join(path, '{}.json'.format(idx))) \
            for idx in indices}


def sort_by_scene_metadata(path, indices, key):
    '''
    Returns an array of indices that is sorted by the specified key in the scene metadata.
    '''
    metadata = read_metadata(path, indices)
    return sorted(indices, key=lambda index: metadata[index]['scene'][key])


def filter_by_scene_metadata(path, indices, key, value_min, value_max):
    '''
    Filters a set of indices by metadata value range
    '''
    metadata = read_metadata(path, indices)
    return sorted([x for x in indices if metadata[x]['scene'][key] >= value_min and \
                   metadata[x]['scene'][key] <= value_max])


def metadata_values(path, indices, key):
    '''
    Extracts the metadata values for the specified indices
    '''
    metadata = read_metadata(path, indices)
    return [metadata[index]['scene'][key] for index in indices]


def euler_angles_to_rotation_matrix(theta):
    Rx = np.array([[1,         0,                  0                   ],
                   [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                   [0,         math.sin(theta[0]), math.cos(theta[0])  ]
    ])
    Ry = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                   [0,                     1,      0                   ],
                   [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
    ])
    Rz = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                   [math.sin(theta[2]),    math.cos(theta[2]),     0],
                   [0,                     0,                      1]
    ])
    R = np.dot(Rz, np.dot(Ry, Rx))
    R4 = np.identity(4)
    R4[:3,:3] = R
    return np.asmatrix(R4)


def translation_matrix(t):
    M = np.identity(4)
    M[:3, 3] = t[:3]
    return np.asmatrix(M)


def vehicle_to_sensor_transform(metadata):
    rpy = (math.radians(metadata['camera']['extrinsic']['roll']),
           math.radians(metadata['camera']['extrinsic']['pitch']),
           math.radians(metadata['camera']['extrinsic']['yaw']))
    t = (metadata['camera']['extrinsic']['x'],
         metadata['camera']['extrinsic']['y'],
         metadata['camera']['extrinsic']['z'])
    R = euler_angles_to_rotation_matrix(rpy)
    T = translation_matrix(t)
    TR = np.dot(T, R)
    return np.linalg.inv(TR)


def vehicle_space_to_image_space(vs_p, vehicle_to_sensor_xform, fx, fy, u0, v0):
    # Sensor coordinates
    scm = np.dot(vehicle_to_sensor_xform, (vs_p[0], vs_p[1], vs_p[2], 1.0))
    sc = (scm[0,0], scm[0,1], scm[0,2])
    # Image coordinates
    u = -sc[1] * fx / sc[0] + u0
    v = -sc[2] * fy / sc[0] + v0
    d = sc[0]
    return (u, v, d)


def vec_sum(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


