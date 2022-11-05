import numpy as np
import tensorflow as tf

def main():
    args = sys.argv
    data_file_name = args[1]
    out_name = args[2]

    data = np.load(data_file_name, allow_pickle=True)
    # data = data.astype(float)
    x_data = data[:,:162]
    t_data = data[:,-2:]
    t_data_human = data[:, 162:164]

    
