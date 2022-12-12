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
        self.legal_move = np.zeros([9, 9], dtype=int)
        self.move_dict = move_dict
        self.move_forward = 0
        # self.full_move = np.zeros([17,17],dtype=bool)
    def promote(self):
        self.name = sepf.promete_name

    def captured(self):
        self.owner = not self.owner
        self.position = [0, 0]

    def capture(self,target):
        target.captured()

    def check_promote():
        return self.name==self.promote_name

    def _set_position(self, position):
        self.position = position

    def _set_legal_move_board(self,board_data):

        board_data_array = np.array(list(board_data.values()))
        print(board_data_array)
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)
        print(s_positions_board)


        move = self.move_dict[self.name].copy()
        # positions = positions[positions[:,0]!=0]
        # o_positions = o_positions[o_positions[:,0]!=0]
        # positions = positions - self.position + 8
        # o_positions = o_positions - self.position + 8

        if not self.owner:
            move = move[::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        # print(move)
        self.legal_move[...] = move[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move[...] *= s_positions_board.astype(int)


    def _set_legal_move_komadai(self,board_data):


        board_data_array = np.array(list(board_data.values()))
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        self.legal_move[...] = 1
        if self.owner:
            self.legal_move[:,self.move_forward:] = 0
        else:
            self.legal_move[:,:-self.move_forward] = 0

        self.legal_move[...] *= (s_positions_board * g_positions_board).astype(int)




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

        board_data_array = np.array(list(board_data.values()),dtype=int)
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if not self.owner:
            move = move[::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        a = np.zeros((17,17), dtype=int)



        b0 = np.array([True,True,True,True],dtype=int)
        s_positions = [tuple(p) for p in np.array(np.where(s_positions_board),dtype=int).T]
        g_positions = [tuple(p) for p in np.array(np.where(g_positions_board),dtype=int).T]
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


        self.legal_move[...] = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= s_positions_board.astype(int)


class KAKU(Piece):

    def __init__(self, init_position, owner, name="KA"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "UM"

    def _set_legal_move_board(self,board_data):
        move = self.move_dict[self.name].copy()
        board_data_array = np.array(list(board_data.values()),dtype=int)
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if not self.owner:
            move = move[::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        a = np.zeros((17,17), dtype=int)



        b0 = np.array([True,True,True,True],dtype=int)
        s_positions = [tuple(p) for p in np.array(np.where(s_positions_board),dtype=int).T]
        g_positions = [tuple(p) for p in np.array(np.where(g_positions_board),dtype=int).T]


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
            # positions_board = np.zeros([17,17],dtype=int)
            # positions_board[position[0],position[1]] = True

            a = a + move[1]
        self.legal_move[...] = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= s_positions_board

class FU(Piece):

    def __init__(self, init_position, owner, name="FU"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "TO"
        self.move_forward = 1

    def _set_legal_move_komadai(self,board_data):
        fu_positions_board = board_data["FU"][1-self.owner][1:,1:]
        board_data_array = np.array(list(board_data.values()))
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        for i in range(len(fu_positions_board)):
            if 1 in fu_positions_board[i]: fu_positions_board[i]=np.zeros_like(fu_positions_board[i],dtype=int)







        self.legal_move[...] = 1
        if self.owner:
            self.legal_move[:,self.move_forward:] = 0
        else:
            self.legal_move[:,:-self.move_forward] = 0

        self.legal_move *= (s_positions_board * g_positions_board * fu_positions_board).astype(int)


class KYO(Piece):

    def __init__(self, init_position, owner, name="KY"):
        super().__init__(init_position, owner,name=name)
        self.promote_name = "NY"
        self.move_forward = 1


    def _set_legal_move_board(self,board_data):
        move = self.move_dict[self.name].copy()
        board_data_array = np.array(list(board_data.values()))
        s_positions_board = np.sum(board_data_array[:, 0, 1:, 1:],axis=0)
        g_positions_board = np.sum(board_data_array[:, 1, 1:, 1:],axis=0)

        if not self.owner:
            move = move[::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        a = np.zeros((17,17), dtype=int)



        b0 = np.array([True,True,True,True],dtype=int)
        s_positions = [tuple(p) for p in np.array(np.where(s_positions_board)).T]
        g_positions = [tuple(p) for p in np.array(np.where(g_positions_board)).T]
        if self.name=="KY":
            if self.owner:
                start,stop = 1, 9
            else:
                start,stop = -1, -9
            for i in range(start,stop):
                x0 = (8,i+8)
                a[x0] = b0[0] * (x0 not in s_positions)
                b0[0] = b0[0] * (x0 not in g_positions)
                # print(b0[0])
                # print(a[x0[0],0])
                b0[0] *= a[x0]

            # a = np.sum(a, axis=0)
        else:


            a = move * s_positions_board


        self.legal_move[...] = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]

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
        self.pieces = np.array(pieces)
        self.turn = 1
        self.board_shape = (9,9)
        self.komadai_mask = [p.position==[0,0] for p in self.pieces]
        self.array_name = np.zeros(len(pieces),dtype="<U2")
        self.array_position = np.zeros([len(pieces),2])
        # print(self.array_position)
        self.array_owner = np.zeros(len(pieces),dtype=int)
        self.array_rawname = np.zeros(len(pieces),dtype="<U2")
        self.array_promote_name = np.zeros(len(pieces),dtype="<U2")
        self.out_data = np.zeros(165)
        self.players = players
        self.raw_names = ["OU", "HI", "KA", "KI", "GI", "KE", "KY", "FU"]
        self.promote_names = ["RY", "UM", "NG", "NK", "NY", "TO"]
        self.promote_names_dict = list(zip(["RY", "UM", "NG", "NK", "NY", "TO"],["HI","KA","GI","KE","KY","FU"]))
        self.names_dict = dict(zip(
        ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
        ["OU", "HI", "HI", "KA", "KA", "KI", "GI", "GI", "KE", "KE", "KY", "KY", "FU", "FU"]
        ))
        self.names = self.raw_names + self.promote_names
        # self.raw_names_dict = {v: k for k, v in self.promote_names_dict.items()}

        # self.positions_board = np.zeros(self.board_shape)


    # def get_positions(self):
    #     self.positions = [p.position for p in self.pieceis]

    def _init_positions(self):
        self.positions_board = np.zeros([2,9,9],dtype=int)
        self.pieces_board = np.zeros([9,9],dtype=int)
        self.pieces_board[:,:] = None
        for p in self.pieces:
            self.positions_board[p.owner,p.position[0],p.position[1]] = True
            self.pieces_board[p.position[0],p.position[1]] = p

    def _init_legal_move(self,piece):
        cross = np.zeros_like(piece.full_move)
        cross[0,] = piece.full_move & self.positions_board[0]
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
        self.array_position[:] = np.array(list(map(self.get_position, self.pieces)))

    def get_array_name(self):
        self.array_name[:] = np.array(list(map(self.get_name, self.pieces)))

    def get_array_rawname(self):
        self.array_rawname[:] = np.array(list(map(self.get_rawname, self.pieces)))

    def get_array_promote_name(self):
        self.array_promote_name[:] = np.array(list(map(self.get_promote_name, self.pieces)))

    def get_array_owner(self):
        self.array_owner[:] = np.array(list(map(self.get_owner, self.pieces)))

    def get_array(self):
        arrays = np.array([[p.position[:], p.name, p.owner, p.rawname, p.promote_name] for p in self.pieces],dtype=object).T

        self.array_position[:] = list(arrays[0])
        self.array_name[:], self.array_owner[:], self.array_rawname[:], self.array_promote_name[:] = arrays[1:]



    def move(self, start, goal, name):
        # self.get_array_name()
        # self.get_array_owner()
        # self.get_array_rawname()
        # self.get_array_promote_name()
        # self.get_array_position()

        self.get_array()

        mask_position = (self.array_position[:,0] == start[0]) * (self.array_position[:,1] == start[1])
        mask_rawname = self.array_rawname == name
        mask_promote_name = self.array_promote_name == name
        p = self.pieces[(mask_rawname + mask_promote_name) * mask_position][0]
        p.position = goal

        if ((self.array_position[:,0]==goal[0])*(self.array_position[:,1]==goal[1])).any():
            p.capture(self.pieces[(self.array_position[:,0]==goal[0])*(self.array_position[:,1]==goal[1])][0])

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

    def board_from_data(self, data):
        self.clear()
        self.get_array()
        for n in self.promote_names:
            d = data[n]
            a = np.array(list(np.where(d))).T
            nn = self.promote_names_dict[n]
            name_mask = self.names == nn
            target_p = self.pieces[name_mask]
            for i,x in enumerate(a):
                p = target_p[i]
                p.name = n
                p.owner = 1 - a[i, 0]
                p.position = tuple(a[i, 1:])

        for n in self.raw_names:
            d = data[n]
            a = np.array(list(np.where(d))).T
            name_mask = self.names == n
            target_p = self.pieces[name_mask]
            for i,x in enumerate(a):
                p = target_p[i]
                p.name = n
                p.owner = 1 - a[i, 0]
                p.position = tuple(a[i, 1:])



    def move_data_tmp(self, start, goal, name):
        data = self.out_data.copy()

        if start[0] == 0:
            d[name][1-self.turn][tuple(goal)] = 1
            data[name][1-self.turn][0,0] -= 1


        else:
            for n in self.names:
                d = data[n]
                # print(d)

                if d[1-self.turn][tuple(start)]:
                    d[1-self.turn][tuple(start)] = 0
                    d[1-self.turn][tuple(goal)] = 1

                if d[self.turn][tuple(goal)]:
                    d[self.turn][tuple(goal)] = 0

                    data[self.names_dict[n]][1-self.turn][0,0] += 1

        return data










    def board_data(self):
        names = ["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"]

        data = np.zeros([14,2,10,10])
        self.out_data = dict(zip(names, data))

        for p in self.pieces:
            self.out_data[p.name][int(p.owner)][p.position] += 1




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
                out = np.array(out)
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

        players, move_data = files.read_csa_file(filename)
        # print(move_data)
        self.players = players

        good = []

        for mv in move_data:
            # print(mv)
            start = [int(mv[0]),int(mv[1])]
            goal = [int(mv[2]),int(mv[3])]
            name = mv[4:6]
            self.board_data()
            moves = self.get_legal_moves()
            a = [list(self.move_data_tmp([int(m[0]),int(m[1])],goal = [int(m[2]),int(m[3])],name=m[4:6]).values()) for m in moves]
            a = np.array(a)


            aa = np.zeros([len(a),2,14,2,10,10])

            # answer = self.out_data.copy()
            aa[:,0,...] = np.array(list(self.out_data.copy().values()))
            aa[:,1,...] = a
            print(aa.shape)
            good.append((np.array(moves) == mv[1:7]))
            self.move(start, goal, name)

            self.turn = 1- self.turn

        out = np.concatenate(aa,axis=0)
        good = np.concatenate(good,axis=0)

        return aa, good




    def get_legal_moves(self):
        moves = []
        self.get_array()
        self.board_data()
        # print(self.out_data)
        for p in self.pieces[:2]:
            if p.owner == self.turn:
                print(p.name)
                p.set_legal_move(self.out_data)
                move_int = list(zip(*np.where(p.legal_move)))
                print(move_int)

                for mi in move_int:
                    if (p.owner * (2*mi[0] - 10) + 10 - mi[0] <= p.move_forward):
                        moves.append(str(p.position[0])+str(p.position[1])+str(mi[0]+1)+str(mi[1]+1)+p.name)
                    if (p.name in self.raw_names) and (p.position[0] != 0) :
                        if (p.owner * (2*mi[0] - 10) + 10 - mi[0] <= 3) or (p.owner * (2*p.position[0] - 10) + 10 - p.position[0] <= 3):
                            moves.append(str(p.position[0])+str(p.position[1])+str(mi[0]+1)+str(mi[1]+1)+p.promote_name)

        return moves








        # pieceis = [OU(init_posision=[4,8], owner=True),
        #            OU(init_posision=[4,0], owner=False),
        #            ]






    # def fname(arg):
    #     pass
