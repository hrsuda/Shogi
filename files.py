import os
import sys
import numpy as np

def get_file_names(dir_path, file_type="csa"):
    files = os.listdir(dir_path)
    out = [n for n in files if file_type == n[-3:]]
    return out
