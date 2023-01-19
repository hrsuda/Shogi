import numpy as np
import pickle
import files
with open('moves.pkl', 'rb') as f:
    MOVE_DICT = pickle.load(f)



# import scipy as sp



class Piece:
    def __init__(self, init_position, owner, move_dict=MOVE_DICT, name="hoge"):
        self.name = name
        self.rawname = name
        self.owner = owner
        self.position = init_position
        self.promote_name = "nr"
        self.legal_move = np.zeros([9, 9], dtype=np.int8)
        self.move_dict = move_dict
        self.move_forward = 0
        self.names = np.array(["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"])

        # self.full_move = np.zeros([17,17],dtype=bool)
    def promote(self):
        self.name = sepf.promete_name

    def captured(self):
        self.owner = 1- self.owner
        self.name = self.rawname
        self.position = (0, 0)

    def capture(self,target):
        target.captured()

    def check_promote():
        return self.name==self.promote_name

    def _set_position(self, position):
        self.position = position

    def _set_legal_move_board(self,board_data):

        board_data_array = board_data.copy()

        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)
        # print(g_positions_board)


        move = self.move_dict[self.name].copy()
        # positions = positions[positions[:,0]!=0]
        # o_positions = o_positions[o_positions[:,0]!=0]
        # positions = positions - self.position + 8
        # o_positions = o_positions - self.position + 8

        if self.owner==0:
            move = move[::-1,::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        # print(move)
        self.legal_move = move[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        # print(self.legal_move)
        self.legal_move *= 1-s_positions_board.astype(np.int8)


    def _set_legal_move_komadai(self,board_data):


        board_data_array = board_data.copy()
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        self.legal_move = np.ones([9,9])

        self.legal_move[...] = ((1-s_positions_board) * (1-g_positions_board)).astype(np.int8)

        if self.owner==1:
            self.legal_move[:,:self.move_forward] = 0
        else:
            self.legal_move[:,9-self.move_forward:] = 0



    def set_legal_move(self, board_data):

        if self.position[0]==0:
            self._set_legal_move_komadai(board_data)
        else:
            self._set_legal_move_board(board_data)




class HISHA(Piece):

    def __init__(self, init_position, owner, name="HI"):
        super().__init__(init_position, owner, name=name)

        self.promote_name = "RY"

        # self.set_legal_move()
        # self.full_move_axis = np.zeros((4,17,17),dtype=bool)
        # self.full_move_axis[0,8,0:8] = True
        # self.full_move_axis[1,8,9:16] = True
        # self.full_move_axis[2,0:8,8] = True
        # self.full_move_axis[3,9:16,8] = True
        # self.full_move = np.zeros((17,17),dtype=bool)
        # self.full_move[8,:] = True
        # self.full_move[:,8] = True
        # self.full_move[8,8] = False
        # self.move_axis = np.zeros((2,4,17,17),dtype=bool)
        # self.move = np.zeros((2,17,17),dtype=bool)

        # self.move_dict["HI"] = np.zeros([4,17,17], dtype=bool)
        # self.move_dict["HI"][0,0:8,0] = True
        # self.move_dict["HI"][1,0,0:8] = True
        # self.move_dict["HI"][2,17:8,0] = True
        # self.move_dict["HI"][3,0,17:8] = True


    #     o_positions = o_positions - self.position + np.array([8,8])
    #     o_positions = o_positions.T
    #
    #     cross = self.full_move_axis[0]

    def _set_legal_move_board(self,board_data):
        move = self.move_dict[self.name].copy()

        board_data_array = board_data.copy()
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if not self.owner:
            # move = move[::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        a = np.zeros((17,17), dtype=np.int8)



        b0 = np.array([True,True,True,True],dtype=np.int8)
        s_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(s_positions_board),dtype=np.int8).T+1]
        g_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(g_positions_board),dtype=np.int8).T+1]
        # print(positions)
        for i in range(1,9):
            x0 = (i+8,8)
            x1 = (-i+8,8)
            x2 = (8,i+8)
            x3 = (8,-i+8)

            a[x0] = b0[0] * (x0 not in s_positions)
            a[x1] = b0[1] * (x1 not in s_positions)
            a[x2] = b0[2] * (x2 not in s_positions)
            a[x3] = b0[3] * (x3 not in s_positions)
            b0[0] = b0[0] * (x0 not in g_positions) * a[x0]
            b0[1] = b0[1] * (x1 not in g_positions) * a[x1]
            b0[2] = b0[2] * (x2 not in g_positions) * a[x2]
            b0[3] = b0[3] * (x3 not in g_positions) * a[x3]
        # print(a)
        # print(b0)

        if self.name == "RY":

            # o_positions_board[o_positions[0],o_positions[1]] = True

            a = a + move[1]


        self.legal_move = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1-s_positions_board.astype(np.int8)


class KAKU(Piece):

    def __init__(self, init_position, owner, name="KA"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "UM"

    def _set_legal_move_board(self,board_data):
        move = self.move_dict[self.name].copy()
        board_data_array = board_data.copy()
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if self.owner == 0:
            # move = move[0][::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        a = np.zeros((17,17), dtype=np.int8)


        b0 = np.array([1,1,1,1],dtype=np.int8)
        s_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(s_positions_board),dtype=np.int8).T+1]
        g_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(g_positions_board),dtype=np.int8).T+1]

        for i in range(1,9):
            x0 = (i+8,i+8)
            x1 = (-i+8,i+8)
            x2 = (i+8,-i+8)
            x3 = (-i+8,-i+8)
            a[x0] = b0[0] * (x0 not in s_positions)
            a[x1] = b0[1] * (x1 not in s_positions)
            a[x2] = b0[2] * (x2 not in s_positions)
            a[x3] = b0[3] * (x3 not in s_positions)
            b0[0] = b0[0] * (x0 not in g_positions) * a[x0]
            b0[1] = b0[1] * (x1 not in g_positions) * a[x1]
            b0[2] = b0[2] * (x2 not in g_positions) * a[x2]
            b0[3] = b0[3] * (x3 not in g_positions) * a[x3]


        if self.name == "UM":
            a = a + move[1]
            # positions_board = np.zeros([17,17],dtype=int)
            # positions_board[position[0],position[1]] = True

        self.legal_move[...] = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1-s_positions_board.astype(np.int8)

class FU(Piece):

    def __init__(self, init_position, owner, name="FU"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "TO"
        self.move_forward = 1

    def _set_legal_move_komadai(self,board_data):
        board_data_array = board_data.copy()
        fu_positions_board = board_data_array[self.names=="FU"][0][int(1-self.owner)][1:,1:]
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)


        fu_legal = np.ones([9,9])
        for i in range(len(fu_positions_board)):
            if 1 in fu_positions_board[i]:
                fu_legal[i,:] = 0







        self.legal_move[...] = np.ones([9,9])
        if self.owner==1:
            self.legal_move[:,0] = 0
        else:
            self.legal_move[:,8] = 0
            s_positions_board, g_positions_board = g_positions_board, s_positions_board


        self.legal_move *= ((1-s_positions_board) * (1-g_positions_board) * fu_legal).astype(np.int8)


class KYO(Piece):

    def __init__(self, init_position, owner, name="KY"):
        super().__init__(init_position, owner,name=name)
        self.promote_name = "NY"
        self.move_forward = 1


    def _set_legal_move_board(self,board_data):
        move = self.move_dict[self.name].copy()

        board_data_array = board_data.copy()
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if self.owner == 0:
            move = move[::-1,::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board




        s_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(s_positions_board),dtype=np.int8).T+1]
        g_positions = [tuple(8+np.array(p)-np.array(self.position)) for p in np.array(np.where(g_positions_board),dtype=np.int8).T+1]
        # print(positions)
        a = np.zeros((17,17), dtype=np.int8)
        b0 = 1
        s1,s2,s3 = -1,-9,-1
        if self.owner==0:
            s1,s2,s3 = 1,9,1
        for i in np.arange(s1,s2,s3):
            x = (8, i + 8)



            # a[x0] = b0[0] * (x0 not in s_positions)
            # a[x1] = b0[1] * (x1 not in s_positions)
            a[x] = b0 * (x not in s_positions)
            # a[x3] = b0[3] * (x3 not in s_positions)
            # b0[0] = b0[0] * (x0 not in g_positions) * a[x0]
            # b0[1] = b0[1] * (x1 not in g_positions) * a[x1]
            b0 = b0 * (x not in g_positions) * a[x]
            # b0[3] = b0[3] * (x3 not in g_positions) * a[x3]
        # print(a)
        # print(b0)

        if self.name == "NY":

            # o_positions_board[o_positions[0],o_positions[1]] = True

            a = move


        self.legal_move = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1-s_positions_board.astype(np.int8)

class KEIMA(Piece):

    def __init__(self, init_position, owner, name="KE"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "NK"
        self.move_forward = 2

class GIN(Piece):

    def __init__(self, init_position, owner, name="GI"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "NG"

class KIN(Piece):

    def __init__(self, init_position, owner, name="KI"):
        super().__init__(init_position, owner, name=name)

class OU(Piece):

    def __init__(self, init_position, owner, name="OU"):
        super().__init__(init_position, owner, name=name)



class Board:
    def __init__(self,pieces,players=["P1", "P2"]):
        self.pieces = tuple(pieces)
        self.turn = 1
        self.board_shape = (9,9)
        self.komadai_mask = [p.position==[0,0] for p in self.pieces]
        self.array_name = np.zeros(len(pieces),dtype="<U2")
        self.array_position = np.zeros([len(pieces),2])
        # print(self.array_position)
        self.array_owner = np.zeros(len(pieces),dtype=np.int8)
        self.array_rawname = np.zeros(len(pieces),dtype="<U2")
        self.array_promote_name = np.zeros(len(pieces),dtype="<U2")
        self.out_data = np.zeros([14,2,10,10])
        self.players = players
        self.raw_names = ["OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU"]
        self.promote_names = ["RY", "UM", "NG", "NK", "NY", "TO"]
        self.promote_names_dict = dict(zip(["RY", "UM", "NG", "NK", "NY", "TO"],["HI","KA","GI","KE","KY","FU"]))
        # self.raw_names_dict = dict(zip(["HI","KA","GI","KE","KY","FU"], ["RY", "UM", "NG", "NK", "NY", "TO"]))
        self.names_dict = dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        ["OU", "HI", "HI", "KA", "KA", "KI", "GI", "GI", "KE", "KE", "KY", "KY", "FU", "FU"]
        ))
        self.names_dict2 = dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        ["OU", "RY", "RY", "UM", "UM", "KI", "NG", "NG", "NK", "NK", "NY", "NY", "TO", "TO"]
        ))
        self.promote_dict = dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        [False,True,False,True,False,False,True,False,True,False,True,False,True,False]
        ))
        self.names = np.array(["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],dtype=str)

        self.raw_names_dict_ind = dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        [0,1,1,3,3,5,6,6,8,8,10,10,12,12]
        ))

        self.names_ind =  dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        ))

        self.promote_ind = dict(zip(
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13],
        [0,1,1,3,3,5,6,6,8,8,10,10,12,12]
        ))
        # self.raw_names_dict = {v: k for k, v in self.promote_names_dict.items()}

        # self.positions_board = np.zeros(self.board_shape)


    # def get_positions(self):
    #     self.positions = [p.position for p in self.pieceis]

    def _init_positions(self):
        self.positions_board = np.zeros([2,9,9],dtype=np.int8)
        self.pieces_board = np.zeros([9,9],dtype=np.int8)
        self.pieces_board[:,:] = None
        for p in self.pieces:
            self.positions_board[p.owner,p.position[0],p.position[1]] = True
            self.pieces_board[p.position[0],p.position[1]] = p

    def _init_legal_move(self,piece):
        cross = np.zeros_like(piece.full_move)
        cross[0] = piece.full_move & self.positions_board[0]
        cross[1] = piece.full_move & self.positions_board[1]


    def get_position(self, p):
        return p.position

    def get_name(self, p):
        return p.name

    def get_rawname(self, p):
        return p.rawname

    def get_promote_name(self, p):
        return p.promote_name

    def get_owner(self, p):
        return p.owner

    def get_array_position(self):
        self.array_position[:] = np.array(list(map(self.get_position, self.pieces))).copy()

    def get_array_name(self):
        self.array_name[:] = np.array(list(map(self.get_name, self.pieces))).copy()

    def get_array_rawname(self):
        self.array_rawname[:] = np.array(list(map(self.get_rawname, self.pieces))).copy()

    def get_array_promote_name(self):
        self.array_promote_name[:] = np.array(list(map(self.get_promote_name, self.pieces))).copy()

    def get_array_owner(self):
        self.array_owner[:] = np.array(list(map(self.get_owner, self.pieces))).copy()

    def get_array(self):
        arrays = np.array([[p.position[:], p.name, p.owner, p.rawname, p.promote_name] for p in self.pieces],dtype=object).T.copy()

        self.array_position[:] = list(arrays[0])
        self.array_name[:], self.array_owner[:], self.array_rawname[:], self.array_promote_name[:] = arrays[1:]



    def move(self, start, goal, name):
        # print(start,goal,name)
        # self.get_array_name()
        # self.get_array_owner()
        # self.get_array_rawname()
        # self.get_array_promote_name()
        # self.get_array_position()

        self.get_array()
        mask_owner = (self.array_owner == self.turn)
        mask_position = (self.array_position[:,0] == start[0]) & (self.array_position[:,1] == start[1])
        mask_rawname = self.array_rawname == name
        mask_promote_name = self.array_promote_name == name
        p = np.array(self.pieces)[(mask_rawname | mask_promote_name) & mask_position & mask_owner][0]
        p.position = tuple(goal)

        if ((self.array_position[:,0]==goal[0])*(self.array_position[:,1]==goal[1])).any()==1:
            p.capture(np.array(self.pieces)[(self.array_position[:,0]==goal[0])*(self.array_position[:,1]==goal[1])][0])

        if name !=p.name:
            p.name = p.promote_name

    # def board_data(self):
    #     # self.get_array_position()
    #     # self.get_array_name()
    #     # self.get_array_owner()
    #     # self.get_array_rawname()
    #     # self.get_array_promote_name()
    #     # self.get_array_position()
    #     self.get_array()
    #     self.out_data[0] = self.turn
    #     self.out_data[1:41] = self.array_position[:,0]
    #     self.out_data[41:81] = self.array_position[:,1]
    #     self.out_data[81:121] = self.array_owner
    #     self.out_data[121:161] = self.array_promote_name == self.array_name

    def clear(self):
        for p in self.pieces:
            p.position = (0,0)
            p.name = p.rawname
            p.owner = 1

        # self.board_data()

    def board_from_data(self, data):
        self.clear()
        # self.board_data()
        self.get_array()
        data_v = data.copy()
        data_k = self.names
        ind = np.where(data_v)
        # print((ind))

        for i in range(len(ind[0])):
            p = self.pieces[i]
            d = np.array(ind,dtype=np.int8).T[i]
            # print(d)
            p.name = data_k[d[0]]
            p.owner = 1-d[1]
            p.position = tuple(d[2:4])

        self.board_data()
        # for n in self.raw_names:
        #     p_mask = self.array_rawname == n
        #     piece = self.pieces[p_mask]
        #     d = data_v[(data_k==n) | (data_k==self.names_dict2[n])]
        #     print(d.shape)
        #
        #     ind = np.array(np.where(d))
        #     print(ind)
        #     for i in range(len(ind[0])):
        #         p = piece[i]
        #         if ind[0,i]==1:
        #             p.name=p.promote_name
        #         p.position = tuple(ind[2:,i])
        #         p.owner = 1 - ind[1,i]






    def move_data_tmp(self, start, goal, name):
        self.board_data()
        data = self.out_data.copy()
        ind = np.array(np.where(data)).T
        name_i = np.where(self.names==name)[0]
        for i in ind:
            if (i[1] == 1-self.turn) & (i[2] == start[0]) & (i[3] == start[1]) & ((self.names[i[0]]==name) or (self.names[i[0]]==self.names_dict2[name])):
                data[tuple(i)] -= 1

            elif (i[1] == self.turn) & (i[2] == goal[0]) & (i[3] == goal[1]):
                data[tuple(i)] -= 1
                data[self.promote_ind[i[0]], 1-self.turn, 0, 0] += 1


            data[self.names_ind[name], 1-self.turn, goal[0], goal[1]] += 1



        #
        # if start[0] == 0:
        #     data[np.where(self.names==name)[0],int(1-self.turn),goal[0],goal[1]] = 1
        #     data[np.where(self.names==name)[0],int(1-self.turn),0,0] -= 1
        #
        #
        # else:
        #     for n in self.names:
        #         d = data[np.where(self.names==name)[0][0]]
        #         # print(d)
        #
        #         if d[1-self.turn,start[0],start[1]]>=1:
        #             d[1-self.turn,start[0],start[1]] -= 1
        #             d[1-self.turn,goal[0],goal[1]] = 1
        #
        #         if d[self.turn,goal[0],goal[1]]==1:
        #             d[self.turn,goal[0],goal[1]] = 0
        #             data[self.names==self.names_dict[n]][0,1-self.turn,0,0] += 1
        #
        #
        self.board_data()
        return data










    def board_data(self):
        data = np.zeros([14,2,10,10],dtype=np.int8)
        # print(self.out_data)

        for p in self.pieces:

            # print(self.names==p.name)
            data[np.where(self.names==p.name)[0],int(1-p.owner),p.position[0],p.position[1]] += 1

            # print(data[self.names==p.name][0][int(1-p.owner)].shape)

            if data[np.where(self.names==p.name)[0],int(1-p.owner),p.position[0],p.position[1]] == 0:
                raise ValueError

            # print(np.where(self.out_data))
            # print(self.out_data.shape)
        if np.sum(data) != 40:
            raise ValueError
        self.out_data = data.copy()

    def read_file(self, filename, ):
        # data = pf.read_csv(filename, comment="'", header=hoge)
        with open(filename) as f:
            data = f.read().split('\n')
        out = []
        for l in data:
            if len(l)==0:
                continue

            elif l[:2]=="N+":
                self.players[0] = l[2:]
            elif l[:2]=="N-":
                self.players[1] = l[2:]

            # if (len(l)==7) and ((l[0]=="+") or (l[0]=="-")):

            elif (len(l)>6) and ((l[0]=="+") or (l[0]=="-")):
                # print(l)
                start = [int(l[1]),int(l[2])]
                goal = [int(l[3]),int(l[4])]
                name = l[5:7]
                self.move(start, goal, name)
                self.turn = 1 - self.turn
                self.board_data()
                out.append(self.out_data.copy())

            elif l == "%TORYO":
                if len(out)==0:
                    # print("0 turn")
                    return None
                result = len(out)%2
                out = np.array(out,dtype=np.int8)
                out[:,-2+result] = 1
                out[:,-4] = "human" in self.players[0]
                out[:,-3] = "human" in self.players[1]

                # print('OK')
                return np.array(out,dtype=np.int8)

            elif "%" in l:
                # print(l)
                return None

    # def read_file_with_bad(self, filename, ):
    #     # data = pf.read_csv(filename, comment="'", header=hoge)
    #     with open(filename) as f:
    #         data = f.read().split('\n')
    #
    #     l0 = None
    #     out = []
    #     good = []
    #     print(filename)
    #     for l in data:
    #         if len(l)==0:
    #             continue
    #
    #         elif l[:2]=="N+":
    #             self.players[0] = l[2:]
    #         elif l[:2]=="N-":
    #             self.players[1] = l[2:]
    #
    #         # if (len(l)==7) and ((l[0]=="+") or (l[0]=="-")):
    #
    #         elif (len(l)>6) and ((l[0]=="+") or (l[0]=="-")):
    #             # print(l)
    #
    #             start = [int(l[1]),int(l[2])]
    #             goal = [int(l[3]),int(l[4])]
    #             name = l[5:7]
    #             self.board_data()
    #             moves = self.get_legal_moves()
    #             a = [np.array(list(self.move_data_tmp([int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=l[4:6]).values())) for m in moves]
    #             a = np.array(a)
    #             aa = np.zeros([len(a),2,14,2,10,10])
    #
    #             # answer = self.out_data.copy()
    #             aa[:,0] = np.array(list(self.out_data.copy().values()))
    #             aa[:,1] = a
    #             good.append((np.array(moves) == l[1:7]))
    #             self.move(start, goal, name)
    #
    #             self.turn = 1- self.turn
    #
    #
    #         elif l == "%TORYO":
    #             if len(aa)==0:
    #                 print("0 turn")
    #                 return None
    #             print(good)
    #             out = np.concatenate(aa,axis=0)
    #             good = np.concatenate(good,axis=0)
    #
    #
    #             # out[:,-2+result] = 1
    #             # out[:,-4] = "human" in self.players[0]
    #             # out[:,-3] = "human" in self.players[1]
    #
    #             print('OK')
    #
    #             return out,good
    #
    #         elif "%" in l:
    #             print('bad')
    #             return None
    #
    #
    #
    def read_file_with_bad(self, filename, ):
        # data = pf.read_csv(filename, comment="'", header=hoge)

        _ = files.read_csa_file(filename)
        if _ is None:return None
        players, move_data  = _
        if len(move_data) < 20:
            print("pass")
            return None
        # print(move_data)
        self.players = players

        good = []
        out = []
        for i,mv in enumerate(move_data):
            # print(mv)
            self.board_data()
            start = [int(mv[0]),int(mv[1])]
            goal = [int(mv[2]),int(mv[3])]
            name = mv[4:6]
            # self.board_data()
            moves = self.get_legal_moves()
            # print(moves)
            a = [list(self.move_data_tmp(start=[int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=m[4:6])) for m in moves]
            a = np.array(a,dtype=np.int8)
            # print(a.shape)

            aa = np.zeros([len(a),2,14,2,10,10])



            # answer = self.out_data.copy()
            self.board_data()
            # print(np.where(self.out_data==1))
            aa[:,0,...] = self.out_data.copy()
            aa[:,1,...] = a
            # print(aa.shape)
            if i % 2==1:
                aa[:,:,:,:,1:,1:] = aa[:, :,:,:, 9:0:-1, 9:0:-1]
                # aa[:,:,:,:] = aa[:,:,:,::-1]
            # print(moves)
            if not (np.array(moves) == mv[0:6]).any():
                raise ValueError
            good.append((np.array(moves) == mv[0:6]))
            out.append(aa)
            self.move(start, goal, name)

            self.turn = 1- self.turn

        out = np.concatenate(out,axis=0)
        good = np.concatenate(good,axis=0)
        print(filename)
        # print(out.shape)
        return out, good


    def read_file_player(self, filename, playername ):
        # data = pf.read_csv(filename, comment="'", header=hoge)

        _ = files.read_csa_file(filename)
        if _ is None:return None
        players, move_data  = _
        if len(move_data) < 20:
            print("pass")
            return None
        # print(move_data)
        self.players = np.array(players)

        p_ind = np.where(self.players == playername)[0][0]




        good = []
        out = []
        for i,mv in enumerate(move_data):
            # print(mv)
            # self.board_data()
            start = [int(mv[0]),int(mv[1])]
            goal = [int(mv[2]),int(mv[3])]
            name = mv[4:6]
            # self.board_data()
            if i % 2 == p_ind:
                moves = self.get_legal_moves()
                # print(moves)
                a = [list(self.move_data_tmp(start=[int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=m[4:6])) for m in moves]
                a = np.array(a,dtype=np.int8)
                # print(a.shape)

                aa = np.zeros([len(a),2,14,2,10,10])



                # answer = self.out_data.copy()
                self.board_data()
                # print(np.where(self.out_data==1))
                aa[:,0,...] = self.out_data.copy()
                aa[:,1,...] = a
                # print(aa.shape)
                if i % 2==1:
                    aa[:,:,:,:,1:,1:] = aa[:, :,:,:, 9:0:-1, 9:0:-1]
                    aa[:,:,:,:] = aa[:,:,:,::-1]
                # print(moves)
                if not (np.array(moves) == mv[0:6]).any():
                    raise ValueError

                good.append((np.array(moves) == mv[0:6]))
                out.append(aa)
            self.move(start, goal, name)

            self.turn = 1- self.turn

        out = np.concatenate(out,axis=0)
        good = np.concatenate(good,axis=0)
        print(filename)
        # print(out.shape)
        return out, good


    def get_legal_moves(self):
        moves = []
        self.get_array()
        self.board_data()
        # print(self.out_data)
        for p in self.pieces:
            if p.owner == self.turn:
                # print(p.name)
                p.set_legal_move(self.out_data.copy())
                move_int = np.array(np.where(p.legal_move),dtype=np.int8).T + 1
                # print(move_int)

                for mi in move_int:

                    if p.owner == 1:
                        if ((p.position[1] <= 3) | (mi[1] <= 3)) & (self.promote_dict[p.name]) & (p.position[0]!=0) :
                            moves.append(str(p.position[0])+str(p.position[1])+str(mi[0])+str(mi[1])+p.promote_name)
                            # print(p.name)
                        if (mi[1] > p.move_forward) or (p.position[0] != 0):
                            moves.append(str(p.position[0])+str(p.position[1])+str(mi[0])+str(mi[1])+p.name)
                    elif p.owner == 0:
                        if ((p.position[1] >= 7) | (mi[1] >= 7)) & self.promote_dict[p.name] & (p.position[0]!=0):
                            moves.append(str(p.position[0])+str(p.position[1])+str(mi[0])+str(mi[1])+p.promote_name)
                        if ((10-mi[1]) > p.move_forward) or (p.position[0] != 0):
                            moves.append(str(p.position[0])+str(p.position[1])+str(mi[0])+str(mi[1])+p.name)


        return np.unique(moves)








        # pieceis = [OU(init_posision=[4,8], owner=True),
        #            OU(init_posision=[4,0], owner=False),
        #            ]






    # def fname(arg):
    #     pass
