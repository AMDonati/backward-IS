import argparse
from train.utils import write_to_csv
import numpy as np
import os

def get_parser():
    parser = argparse.ArgumentParser()
    # data parameters:
    parser.add_argument("-folder_path", type=str, required=True, help="path for uploading the results")
    return parser