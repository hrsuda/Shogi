aimport numpy as np
import pandas as pd
# import scipy as sp



class Piece():
    def __init__(self, init_position, owner, name="hoge"):
        self.name = name
        self.rawname = name
        self.owner = owner
        self.position = init_posision
        self.promote_name = "nari"

        # self.full_move = np.zeros([17,17],dtype=bool)
    def promote(self):
        self.name = sepf.promete_name

    def captured(self):
        self.owner = ~self.owner
        self.position = [9, 9]

    def capture(self,target):
        target.captured()


    def _set_position(self, position):
        self.position = position

class HISHA(Piece):

    def __init__(self,  name="HI"):
        super.__init__(name=name)

        self.promete_name = "RY"
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

    def __init__(self,  name="KA"):
        super.__init__(name=name)
        self.promote_name = "UM"

class FU(Piece):

    def __init__(self,  name="FU"):
        super.__init__(name=name)
        self.promote_name = "TO"
class KYO(Piece):

    def __init__(self,  name="KY"):
        super.__init__(name=name)
        self.promote_name = "NY"

class KEIMA(Piece):

    def __init__(self,  name="KE"):
        super.__init__(name=name)
        self.promote_name = "NK"

class GIN(Piece):

    def __init__(self, name="GI"):
        super.__init__(name=name)
        self.promote_name = "NG"

class KIN(Piece):

    def __init__(self,  name="KI"):
        super.__init__(name=name)

class OU(Piece):

    def __init__(self,  name="OU"):
        super.__init__(name=name)



class Board():
    def __init__(self,pieces):
        self.pieces = tuple(pieces)
        self.turn = False
        self.board_shape = (9,9)
        self.komadai_mask = [p.position==[9,9] for p in self.pieces]
        self.array_name = np.zeros(len(pieces))
        self.array_position = np.zeros(len(pieces))
        self.array_owner = np.zeros(len(pieces))
        self.array_rawname = np.zeros(len(pieces))
        self.array_promote_name = np.zeros(len(pieces))
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
        self.array_position[:] = np.array(map(self.get_position, self.pieces))

    def get_arrat_name(self):
        self.array_name[:] = np.array(map(self.get_name, self.pieces))

    def get_array_rawname(self):
        self.array_rawname[:] = np.array(map(self.get_rawname, self.pieces))

    def get_array_promote_name(self):
        self.array_promote_name[:] = np.array(map(self.get_promote_name, self.pieces))

    def get_array_owner(self):
        self.array_owner[:] = np.array(map(self.get_owner, self.pieces))

    def move(self, start, goal, name):
        mask_position = self.array_position == start
        mask_rawname = self.array_rawname == name
        mask_promote_name = self.array_promote_name == name
        p = self.pieces[(mask_rawname | mask_promote_name) & mask_position][0]
        p.position = goal

        if any(self.array_position==goal):
            p.capture(self.pieceis[self.array_position==goal])

        if name!=p.name:
            p.name = promote_name

    def board_data(self):
        self.get_array_position()

        self.out_data[0] = self.turn
        self.out_data[1:41] = self.array_position[:,0]*9 + self.array_position[:,1]
        self.out_data[41:82] = self.array_owner
    def read_file(self, filename):
        # data = pf.read_csv(filename, comment="'", header=hoge)
        with open(filename) as f:
            data = f.read().split('\n')
        out = []
        for l in data:
            if l[0] in [0,1,2,3,4,5,6,7,8,9]:
                start = [int(l[0]),int(l[1])]
                goal = [int(l[2]),int(l[3])]
                name = l[4:6]
                self.move(start, goal, name)
                self.turn = ~self.turn
                out.append(self.board_data())

            elif l == "%TORYO":
                result = len(out)%2==1
                out.append(result)

            return out


        # pieceis = [OU(init_posision=[4,8], owner=True),
        #            OU(init_posision=[4,0], owner=False),
        #            ]






    # def fname(arg):
    #     pass
