import numpy as np
import pandas as pd
# import scipy as sp



class Piece:
    def __init__(self, init_position, owner, name="hoge"):
        self.name = name
        self.rawname = name
        self.owner = owner
        self.position = np.array(init_position)
        self.promote_name = "nr"

        # self.full_move = np.zeros([17,17],dtype=bool)
    def promote(self):
        self.name = sepf.promete_name

    def captured(self):
        self.owner = ~self.owner
        self.position = [0, 0]

    def capture(self,target):
        target.captured()

    def check_promote():
        return self.name==self.promote_name
    def _set_position(self, position):
        self.position = position

class HISHA(Piece):

    def __init__(self, init_position, owner, name="HI"):
        super().__init__(init_position, owner, name=name)

        self.promote_name = "RY"
        # self.set_legal_move()
        self.full_move_axis = np.zeros((4,17,17),dtype=bool)
        self.full_move_axis[0,8,0:8] = True
        self.full_move_axis[1,8,9:16] = True
        self.full_move_axis[2,0:8,8] = True
        self.full_move_axis[3,9:16,8] = True
        self.full_move = np.zeros((17,17),dtype=bool)
        self.full_move[8,:] = True
        self.full_move[:,8] = True
        self.full_move[8,8] = False
        self.move_axis = np.zeros((2,4,17,17),dtype=bool)
        self.move = np.zeros((2,17,17),dtype=bool)


    def set_legal_move(self,positions,o_positions):
        positions = positions - self.position + np.array([8,8])
        positions = positions.T
        o_positions = o_positions - self.position + np.array([8,8])
        o_positions = o_positions.T

        cross = self.full_move_axis[0]

        # self.full_move = np.array([[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
        #                            [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]])

        # self.full_move[8,:] = True
        # self.full_move[:,8] = True
        # self.full_move[8,8] = False
        # self.legal_move_nopieces = self.full_move[9-self.position[0]:18-self.position[0],

                                             # 9-self.position[1]:18-self.position[1]]
class KAKU(Piece):

    def __init__(self, init_position, owner, name="KA"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "UM"

class FU(Piece):

    def __init__(self, init_position, owner, name="FU"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "TO"
class KYO(Piece):

    def __init__(self, init_position, owner, name="KY"):
        super().__init__(init_position, owner,name=name)
        self.promote_name = "NY"

class KEIMA(Piece):

    def __init__(self, init_position, owner, name="KE"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "NK"

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
    def __init__(self,pieces):
        self.pieces = np.array(pieces)
        self.turn = False
        self.board_shape = (9,9)
        self.komadai_mask = [p.position==[0,0] for p in self.pieces]
        self.array_name = np.zeros(len(pieces),dtype="<U2")
        self.array_position = np.zeros([len(pieces),2])
        # print(self.array_position)
        self.array_owner = np.zeros(len(pieces),dtype=bool)
        self.array_rawname = np.zeros(len(pieces),dtype="<U2")
        self.array_promote_name = np.zeros(len(pieces),dtype="<U2")
        self.out_data = np.zeros(163)
        # self.positions_board = np.zeros(self.board_shape)


    # def get_positions(self):
    #     self.positions = [p.position for p in self.pieceis]

    def _init_positions(self):
        self.positions_board = np.zeros([2,9,9],dtype=bool)
        self.pieces_board = np.zeros([9,9],dtype=bool)
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

        if name!=p.name:
            p.name = p.promote_name

    def board_data(self):
        # self.get_array_position()
        # self.get_array_name()
        # self.get_array_owner()
        # self.get_array_rawname()
        # self.get_array_promote_name()
        # self.get_array_position()
        self.get_array()
        self.out_data[0] = self.turn
        self.out_data[1:41] = self.array_position[:,0]
        self.out_data[41:81] = self.array_position[:,1]
        self.out_data[81:121] = self.array_owner
        self.out_data[121:161] = self.array_promote_name == self.array_name

    def read_file(self, filename):
        # data = pf.read_csv(filename, comment="'", header=hoge)
        with open(filename) as f:
            data = f.read().split('\n')
        out = []
        for l in data:
            # if (len(l)==7) and ((l[0]=="+") or (l[0]=="-")):
            if (len(l)>6) and ((l[0]=="+") or (l[0]=="-")):
                # print(l)
                start = [int(l[1]),int(l[2])]
                goal = [int(l[3]),int(l[4])]
                name = l[5:7]
                self.move(start, goal, name)
                self.turn = ~self.turn
                self.board_data()
                out.append(self.out_data.copy())

            elif l == "%TORYO":
                if len(out)==0:
                    # print("0 turn")
                    return None
                result = len(out)%2
                out = np.array(out)
                out[:,-2+result] = 1
                # print('OK')
                return np.array(out,dtype=np.int8)

            elif "%" in l:
                # print(l)
                return None


        # pieceis = [OU(init_posision=[4,8], owner=True),
        #            OU(init_posision=[4,0], owner=False),
        #            ]






    # def fname(arg):
    #     pass
