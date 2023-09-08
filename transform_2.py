from shogi_3 import *
import numpy as np
import files
import pickle
import init_position
import sys


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

    for i,fname in enumerate(file_names):
        if i > 1000:break
        with open("init_posision.pkl", "rb") as f:
            pieces = pickle.load(f)
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

            np.savez_compressed(out_filename + str(i)+"_k", output_k_np.astype(np.int8))
            np.savez_compressed(out_filename + str(i)+"_b", output_b_np.astype(np.int8))
            np.savez_compressed(out_filename + str(i) +'_t', output_g_np.astype(np.int8))
            output_b = []
            output_k = []
            output_g = []



    output_b_np = np.concatenate(output_b, axis=0)
    output_k_np = np.concatenate(output_k, axis=0)
    output_g_np = np.concatenate(output_g, axis=0)
    # print(np.array(output).shape)

    np.savez_compressed(out_filename + str(i)+"_k", output_k_np.astype(np.int8))
    np.savez_compressed(out_filename + str(i)+"_b", output_b_np.astype(np.int8))
    np.savez_compressed(out_filename + str(i) +'_t', output_g_np.astype(np.int8))

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
