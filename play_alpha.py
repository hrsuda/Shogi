import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import datetime
import os

from shogi_3 import *
import learning_numpy as learning_numpy
import plot_shogi as ps


def main():
    plt.ion()
    args = sys.argv

    save_dir_path = "../game_log/" + datetime.datetime.now().strftime("%Y%m%d/")
    save_file_name = datetime.datetime.now().strftime("%Y%m%d%H%M")
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)



    network_data_filename = args[1]
    if network_data_filename[-4:] == ".pkl":
        with open(network_data_filename, "rb") as f:
            network = pickle.load(f)
    elif network_data_filename[-4:] == ".csv":
        network = learning_numpy.ShogiLayerNet3()
        W_params = np.loadtxt(network_data_filename)
        b_params = np.loadtxt(network_data_filename.replace("_W.csv","_b.csv"))
    

    # i_n = [2*2*14*33, 2*2*14*33, 2*2*14*5, 2*2*14*9, (81+81+81+81+14*2*2)*100, 1,
    #         4*33, 4*33, 4*5, 4*9, (81+81+81+81+100)*50, 1,
    #         4*33, 4*33, 4*5, 4*9, 81+81+81+81+50]
    # # i_b = [1, 1, 1, 1, 100, 1, 1, 1, 1, 1, 50, 1, 1, 1, 1, 1, 1]
    # i_b = [81, 81, 81, 81, 100, 1,
    #         81, 81, 81, 81, 50, 1,
    #         81, 81, 81, 81, 1]
    # network.W_params[0] = W_params[0:2*2*14*33]
    # network.b_params[0] = b_params[0:1]   

    # network.W_params[1] = W_params[2*2*14*33:2*2*2*14*33]
    # network.b_params[1] = b_params[1:2]   

    # network.W_params[2] = W_params[2*2*2*14*33:2*2*2*14*33+2*2*14*3*3].reshape([2*2*14,3,3])
    # network.b_params[2] = b_params[2:3]   

    # network.W_params[3] = W_params[2*2*2*14*33+2*2*14*3*3:2*2*2*14*33+2*2*14*3*3+81+81+81+14*2*2].reshape(81+81+81+14*2*2,1)
    # network.b_params[3] = b_params[3:4]

    i_0 = 0
    i_0b = 0   
    for i in range(len(network.layers)):
        l = network.layers[i]

        i_w = l.W.size
        i_b = l.b.size
        
        l.W = W_params[i_0:i_0+i_w].reshape(l.W.shape)
        l.b = b_params[i_0b:i_0b+i_b].reshape(l.b.shape)
        print(l.W.shape)
        print(l.b.shape)
        # network.W_params[i] = W_params[i_0:i_0+i_n[i]]
        # network.b_params[i] = b_params[i_0b:i_0b+i_b[i]]   
        i_0 = i_0+i_w
        i_0b = i_0b+i_b
        

    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    B = Board(pieces)
    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    B_predict = Board(pieces)

    sente = np.random.randint(low=0,high=2)
    teban = sente
    players = ["human", "sudanza"]
    game = True
    
    out_b = []
    out_k = []
    out_t = []
    while game:
        print(teban)
        plt.cla()
        ax = ps.plot_board_alpha(B.pieces)
        plt.show()

        moves = B.get_legal_moves()
        move_data_b = []
        move_data_k = []
        if B.turn==0:
            for m in moves:
                b,k = B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:])
                move_data_b.append(b)
                move_data_k.append(k)
        else:
            for m in moves:
                # move_data.append(B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:])[:,::-1,::-1,::-1])
                b,k = B.move_data_tmp((int(m[0]),int(m[1])),(int(m[2]),int(m[3])),m[4:])
                move_data_b.append(b[::-1,:,::-1,::-1])
                move_data_k.append(k[::-1,:])

        if teban == 0:
            move_str = ''
            while True:
                move_str = input().upper()
                if (move_str in moves) or (move_str in ["TORYO", "END"]):break

            if move_str in ["TORYO", "END"]:
                break

            for md in move_data_b:
                out_b.append(np.array([B.banmen,md]))
            for md in move_data_k:
                out_k.append(np.array([B.komadai,md]))
            for m in moves:
                out_t.append(m==move_str)


        else:
            result = []
            for i in range(len(move_data_b)):
                result.append(learning_numpy.sigmoid(network.predict([np.array([[B.banmen,move_data_b[i]]]),np.array([[B.komadai,move_data_k[i]]])])))
            ind = np.argsort(result,axis=0,)[::-1,0,0]
            # move_str = moves[ind[0]]
            inds = [ii for ii, v in enumerate(result) if v == max(result)]
            inds = np.random.choice(inds,1)[0]
            move_str = moves[inds]
            for i in range(20):
                print(str(result[ind[i]]*100) + ' ' + moves[ind[i]] )

            # result = []
            # for i in range(len(move_data_b)):
            #     next_b = move_data_b[i]
            #     next_k = move_data_k[i]

            #     mv_point_1 = learning_numpy.sigmoid(network.predict([np.array([[B.banmen,next_b]]),np.array([[B.komadai,next_k]])]))
            #     B_predict.board_from_data(next_b, next_k)
            #     rm = B_predict.get_legal_moves()
            #     mv_point_max = 0
            #     move_1 = None
            #     for m in rm:
            #         mp = learning_numpy.sigmoid(network.predict([np.array([[B_predict.banmen,next_b]]),np.array([[B_predict.komadai,next_k]])]))
            #         if mv_point_max<mp:
            #             mv_point_max = mp
            #             move_1 = m
            #     B_predict.move_from_string(move=move_1)

            #     rm = B_predict.get_legal_moves()
            #     mv_point_max = 0
            #     move_1 = None
            #     for m in rm:
            #         mp = learning_numpy.sigmoid(network.predict([np.array([[B_predict.banmen,next_b]]),np.array([[B_predict.komadai,next_k]])]))
            #         if mv_point_max<mp+mv_point_1:
            #             mv_point_max = mp + mv_point_1

            #             move_1 = m

            #     result.append(mv_point_max)


            # ind = np.argsort(result,axis=0,)[::-1,0,0]
            # move_str = moves[ind[0]]
            # for i in range(20):
            #     print(str(result[ind[i]]*100) + ' ' + moves[ind[i]] )




        B.move(start=(int(move_str[0]), int(move_str[1])),goal=(int(move_str[2]),int(move_str[3])),name=move_str[4:])
        teban = 1 - teban
        B.turn = 1 - B.turn
    out_b = np.array(out_b,dtype=int)
    out_k = np.array(out_k,dtype=int)
    out_t = np.array(out_t,dtype=int)

    np.savez_compressed(save_dir_path+save_file_name+"_b", out_b)
    np.savez_compressed(save_dir_path+save_file_name+"_k", out_k)

    np.savez_compressed(save_dir_path+save_file_name+'_t', out_t)


if __name__ == "__main__":
    main()
