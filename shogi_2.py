import numpy as np
# import scipy as sp



class Piece():
    def __init__(self,name="hoge"):
        self.name = name
        self.rawname = name
        self.owner = False
        self.position = [0,0]
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

    def __init__(self, init_posision, name="HI"):
        super.__init__(name=name)
        self.position = init_posision

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

    def __init__(self, init_posision, name="KA"):
        super.__init__(name=name)
        self.position = init_posision
        self.promote_name = "UM"

class FU(Piece):

    def __init__(self, init_posision, name="FU"):
        super.__init__(name=name)
        self.position = init_posision
        self.promote_name = "TO"
class KYO(Piece):

    def __init__(self, init_posision, name="KY"):
        super.__init__(name=name)
        self.position = init_posision
        self.promote_name = "NY"

class KEIMA(Piece):

    def __init__(self, init_posision, name="KE"):
        super.__init__(name=name)
        self.position = init_posision
        self.promote_name = "NK"

class GIN(Piece):

    def __init__(self, init_posision, name="GI"):
        super.__init__(name=name)
        self.position = init_posision
        self.promote_name = "NG"

class KIN(Piece):

    def __init__(self, init_posision, name="KI"):
        super.__init__(name=name)
        self.position = init_posision

class OU(Piece):

    def __init__(self, init_posision, name="OU"):
        super.__init__(name=name)
        self.position = init_posision



class Board():
    def __init__(self):
        self.pieces = []
        self.turn = False
        self.board_shape = (9,9)
        self.komadai = np.zeros(40)
        # self.positions_board = np.zeros(self.board_shape)


    # def get_positions(self):
    #     self.positions = [p.position for p in self.pieceis]

    def _init_positions(self):
        self.positions_board = np.zeros([2,9,9],dtype=bool)
        self.pieces_board = np.zeros([9,9],dtype=bool)
        self.pieces_board[:,:] = None
        for p in self.positions:
            self.positions_board[p.owner,p.position[0],p.position[1]] = True
            self.pieces_board[p.position[0],p.position[1]] = p

    def _init_legal_move(self,piece):
        cross = np.zeros_like(piece.full_move)
        cross[0,] = piece.full_move & self.positions_board[0]
        cross[1] = piece.full_move & self.positions_board[1]


    def move(self, start, goal, promote=False):
        if start==[0,0]:
            a = positions_board[goal]
            

    # def fname(arg):
    #     pass
