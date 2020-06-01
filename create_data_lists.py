import argparse
import csv
import glob
import os
import sys
from os.path import split

import matplotlib.pyplot as plt
import pandas as pd

import cv2
import oyaml
from attrdict import AttrDict
from utils import create_data_lists

###################### Load Config File #############################
parser = argparse.ArgumentParser(description='Run training of outlines prediction model')
parser.add_argument('-c', '--configFile', required=True, help='Path to config yaml file', metavar='path/to/config')
args = parser.parse_args()

CONFIG_FILE_PATH = args.configFile
with open(CONFIG_FILE_PATH) as fd:
    config_yaml = oyaml.load(fd)  # Returns an ordered dict. Used for printing

config = AttrDict(config_yaml)

def create_csv(location):
    """In the given location, reads all the subdirs (classes) and files inside them and creates a csv file
        which contains information about the products such as bounding box co-ord, product category

    Args:
        location ([String]): [path to the classes directory]
    """
    for product_category in os.listdir(location):
        if os.path.isdir(os.path.join(location, product_category)):
            for file in os.listdir(os.path.join(location, product_category)):
                file_lst = file.split('_')
                file_name = file_lst[0] + '_' + file_lst[1] + '_' + file_lst[2] + '_' + file_lst[3] + '_' + file_lst[4]
                shelf_id = file_lst[0] + '_' + file_lst[1]
                snapshot = file_lst[2] + '_' + file_lst[3] + '_' + file_lst[4].split('.')[0]
                x_min = int(file_lst[5])
                y_min = int(file_lst[6])
                width = int(file_lst[7])
                height = int(file_lst[8].split('.')[0])
                x_max = x_min + width
                y_max = y_min + height
                with open(os.path.join(location, csv_filename), 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
                    row_data = [file, file_name, shelf_id, snapshot, int(product_category), x_min, y_min, width, height, x_max, y_max]
                    writer.writerow(dict(zip(field_names, row_data)))

if __name__ == '__main__':
    location = config.datapreprocessing.location
    csv_filename = config.datapreprocessing.csv_filename
    field_names = list(config.datapreprocessing.field_names_in_csv)
    with open(os.path.join(location, csv_filename), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names, delimiter=',')
        writer.writeheader()

    # method to create csv file
    create_csv(location)

    create_data_lists(shelf_images_train = config.datapreprocessing.shelf_images_train,
                      shelf_images_val=config.datapreprocessing.shelf_images_val,
                      shelf_images_test = config.datapreprocessing.shelf_images_test,
                      csv_file_with_details = config.datapreprocessing.location + '/' + config.datapreprocessing.csv_filename,
                      output_folder = config.datapreprocessing.output_folder)
