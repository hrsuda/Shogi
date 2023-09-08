from shogi_3 import *
import numpy as np
import files
import pickle
import init_position
import sys
import h5py


def main():
    init_position.main()
    args = sys.argv
    input_dir = args[1]
    out_filename = args[2]

    file_names = files.get_file_names(input_dir, file_type="csa")
    output_b = []
    output_k = []
    output_g = []

    out_0 = None

    f = h5py.File(out_filename+"data_2" + '.hdf5',"a")
    try:
        group = f.create_group('/data')
        h5data_b = group.create_dataset(name='data_b',shape=(0,2,2,14,9,9),chunks = True,maxshape=(None,2,2,14,9,9),dtype=np.int8)
        h5data_k = group.create_dataset(name='data_k',shape=(0,2,2,14),chunks = True,maxshape=(None,2,2,14),dtype=np.int8)
        h5data_t = group.create_dataset(name='data_t',shape=(0,),chunks = True,maxshape=(None,),dtype=np.int8)
    except:
        group = f["/data"]
        h5data_b = group["data_b"]
        h5data_k = group["data_k"]
        h5data_t = group["data_t"]




    for i,fname in enumerate(file_names):
        # if i > 1000:break
        with open("init_posision.pkl", "rb") as fp:
            pieces = pickle.load(fp)
        B = Board(pieces)
        # print(fname)
        # out,good = B.read_file(input_dir+"/"+fname)
        out = B.read_file_with_bad(fname)
        if out is None:continue
        banmen, komadai, good = out
        print(np.where(good))

                  #
        output_b.append(banmen)
        output_k.append(komadai)
        output_g.append(good)



        if (i % 20 == 0) & (i>0):
            print(i)
            output_b_np = np.concatenate(output_b, axis=0)
            output_k_np = np.concatenate(output_k, axis=0)
            output_g_np = np.concatenate(output_g, axis=0)
            # print(np.array(output).shape)

            size = h5data_b.shape[0]
            add_size = output_b_np.shape[0]
            h5data_b.resize((size+add_size,2,2,14,9,9))
            h5data_b[size:] = output_b_np

            h5data_k.resize((size+add_size,2,2,14))
            h5data_k[size:] = output_k_np

            h5data_t.resize((size+add_size,))
            h5data_t[size:] = output_g_np


            # np.savez_compressed(out_filename + str(i)+"_k", output_k_np.astype(np.int8))
            # np.savez_compressed(out_filename + str(i)+"_b", output_b_np.astype(np.int8))
            # np.savez_compressed(out_filename + str(i) +'_t', output_g_np.astype(np.int8))
            output_b = []
            output_k = []
            output_g = []



    output_b_np = np.concatenate(output_b, axis=0)
    output_k_np = np.concatenate(output_k, axis=0)
    output_g_np = np.concatenate(output_g, axis=0)
    # print(np.array(output).shape)

    size = h5data_b.shape[0]
    add_size = output_b_np.shape[0]
    h5data_b.resize((size+add_size,2,2,14,9,9))
    h5data_b[size:] = output_b_np

    h5data_k.resize((size+add_size,2,2,14))
    h5data_k[size:] = output_k_np

    h5data_t.resize((size+add_size,))
    h5data_t[size:] = output_g_np

    f.flush()
    f.close()
    # np.savez_compressed(out_filename + str(i)+"_k", output_k_np.astype(np.int8))
    # np.savez_compressed(out_filename + str(i)+"_b", output_b_np.astype(np.int8))
    # np.savez_compressed(out_filename + str(i) +'_t', output_g_np.astype(np.int8))

# def main():
#     init_position.main()
#     args = sys.argv
#     input_dir = args[1]
#     out_filename = args[2]
#
#     file_names = files.get_file_names(input_dir, file_type="csa")
#     output = []
#     for i,fname in enumerate(file_names):
#         with open("init_posision.pkl", "rb") as f:
#             pieces = pickle.load(f)
#         B = Board(pieces)
#         # print(fname)
#         out = B.read_file(input_dir+"/"+fname)
#
#         if out is not None:
#             # print(i)
#             output.append(out)
#         # if i==1000:
#         #     break
#
#     output = np.concatenate(output)
#     np.save(out_filename, output)

if __name__ == "__main__":
    main()
