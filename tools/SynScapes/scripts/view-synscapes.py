#! /usr/bin/env python3

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt

import helpers

# Arguments ---

scene_metadata = [
    "altitude_variation",
    "curb_height",
    "dist_bicyclists_mean",
    "dist_bicyclists_stddev",
    "dist_buses_mean",
    "dist_buses_stddev",
    "dist_cars_mean",
    "dist_cars_stddev",
    "dist_motorcyclists_mean",
    "dist_motorcyclists_stddev",
    "dist_pedestrians_mean",
    "dist_pedestrians_stddev",
    "dist_trains_mean",
    "dist_trains_stddev",
    "dist_trucks_mean",
    "dist_trucks_stddev",
    "ego_speed",
    "fence_height",
    "fence_presence",
    "median_presence",
    "num_bicyclists",
    "num_buses",
    "num_cars",
    "num_motorcyclists",
    "num_pedestrians",
    "num_trains",
    "num_trucks",
    "parking_angle",
    "parking_presence",
    "rel_dist_to_isect",
    "sidewalk_width",
    "sky_contrast",
    "sun_height",
    "wall_height",
    "wall_presence"
]


parser = argparse.ArgumentParser(
    description='Displays images in the SynScapes dataset with optional sorting by metadata.')

parser.add_argument('path',
                    help='Path to SynScapes root directory.')
parser.add_argument('--analyze-num', '-a', type=int,
                    help='Maximum number of images to analyze')
parser.add_argument('--display-num', '-d', type=int,
                    help='Maximum number of images to display')
parser.add_argument('--sort', '-s', choices=scene_metadata,
                    help='Sort by the specified metadata key.')
parser.add_argument('--restrict', '-r',
                    help='Restricts the choice of image to one or more ')
parser.add_argument('--plot', '-p', action='store_true',
                    help='Plots the values used to sort (requires --sort).')

# DEBUG
parser.add_argument('--meta-dir')

args = parser.parse_args()

# Script ---

root = os.path.abspath(args.path)

# Ensure root exists
if not os.path.exists(root):
    print('Invalid path:', root)
    sys.exit(1)

img_dir, meta_dir = [os.path.join(root, x) for x in ['img', 'meta']]

if args.meta_dir:
    meta_dir = args.meta_dir

# Ensure integrity
for d in [img_dir, meta_dir]:
    if not os.path.exists(d):
        print('Missing directory:', d)
        sys.exit(1)

# By default, consider all files
indices = sorted([int(f.split('.')[0]) for f in os.listdir(meta_dir)])

# Restrict number to analyze
if args.analyze_num:
    indices = indices[::helpers.stride(len(indices), args.analyze_num)]

# Restrict by metadata values
if args.restrict:
    for pair in args.restrict.split(','):
        name, value_range = pair.split('=')
        value_min, value_max = [float(x) for x in value_range.split('-')]
        indices = helpers.filter_by_scene_metadata(meta_dir, indices, name, value_min, value_max)

# Sort
if args.sort:
    print('Reading metadata...')
    indices = helpers.sort_by_scene_metadata(meta_dir, indices, args.sort)

# Restrict number to display
if args.display_num:
    indices = indices[::helpers.stride(len(indices), args.display_num)]
    
# Display
rgb_path = os.path.join(img_dir, 'rgb')
images = [os.path.join(rgb_path, '{}.png'.format(idx)) for idx in indices]
cmd = 'feh {} &'.format(' '.join(images))
os.system(cmd)

# Plot
if args.sort and args.plot:
    y_values = helpers.metadata_values(meta_dir, indices, args.sort)
    # Style
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["bottom"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.spines["left"].set_visible(False)
    ax.get_xaxis().tick_bottom()    
    ax.get_yaxis().tick_left()
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.tick_params(axis="both", which="both", bottom="off", top="off",    
    #            labelbottom="on", left="off", right="off", labelleft="on")
    # Plot
    plt.style.use('seaborn-muted')
    plt.plot(y_values, linewidth=3.0, color=helpers.tableau20[0])
    plt.title(args.sort)
    plt.show()
