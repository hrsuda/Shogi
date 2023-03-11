import numpy as np
import copy
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
        self.owner = 1 - self.owner
        self.name = self.rawname
        self.position = (0, 0)

    def capture(self,target):
        target.captured()

    def check_promote(self, goal):
        bool_name = self.name!=self.promote_name
        if position==(0,0):
            bool_position = False
        elif self.owner==0:
            bool_position = (p.position[1]<4) | (goal[1]<4)
        elif self.owner==1:
            bool_position = (p.position[1]>6) | (goal[1]>6)

        return bool_name & bool_position

    def check_no_promote(self, goal):
        return True

    def _set_position(self, position):
        self.position = position

    def _set_legal_move_board(self, banmen):


        move = self.move_dict[self.name]
        positions = np.sum(banmen[self.owner], axis=0)

        if self.owner==1:
            move = move[::-1,::-1]
            s_positions_board, g_positions_board = g_positions_board, s_positions_board

        self.legal_move = move[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1 - positions


    def _set_legal_move_komadai(self, banmen):


        positions = np.sum(banmen,axis=(0,1))
        self.legal_move = 1 - positions

        if self.owner==0:
            self.legal_move[:,:self.move_forward] = 0
        else:
            self.legal_move[:,9-self.move_forward:] = 0



    def set_legal_move(self, banmen):

        if self.position[0]==0:
            self._set_legal_move_komadai(banmen)
        else:
            self._set_legal_move_board(banmen)




class HISHA(Piece):

    def __init__(self, init_position, owner, name="HI"):
        super().__init__(init_position, owner, name=name)

        self.promote_name = "RY"


    def _set_legal_move_board(self,banmen):
        move = self.move_dict[self.name]

        s_positions, g_positions = np.sum(banmen,axis=1)

        if not self.owner:
            s_positions, g_positions = g_positions, s_positions
        s_positions_17, g_positions_17 = np.zeros((2,17,17))
        s_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = s_positions
        g_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = g_positions
        a = np.zeros((17,17))
        b0 = np.array([True,True,True,True])
        # print(positions)
        for i in range(1,9):
            x = ((i+8,8),
                (-i+8,8),
                (8,i+8),
                (8,-i+8))
            a = b0 * s_positions_17[x[0],x[1],x[2],x[3]]
            b0 = b0 * g_positions_17[x[0],x[1],x[2],x[3]] * a
        if self.name == "RY":
            a = a + move[1]

        self.legal_move = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1-s_positions


class KAKU(Piece):

    def __init__(self, init_position, owner, name="KA"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "UM"

    def _set_legal_move_board(self,banmen):
        move = self.move_dict[self.name]

        s_positions, g_positions = np.sum(banmen,axis=1)

        if not self.owner:
            s_positions, g_positions = g_positions, s_positions
        s_positions_17, g_positions_17 = np.zeros((2,17,17))
        s_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = s_positions
        g_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = g_positions
        a = np.zeros((17,17))
        b0 = np.array([True,True,True,True])
        # print(positions)
        for i in range(1,9):
            x = ((i+8,i+8),
                (-i+8,i+8),
                (i+8,-i+8),
                (-i+8,-i+8))
            a = b0 * s_positions_17[x[0],x[1],x[2],x[3]]
            b0 = b0 * g_positions_17[x[0],x[1],x[2],x[3]] * a
        if self.name == "UM":
            a = a + move[1]

        self.legal_move = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
        self.legal_move *= 1-s_positions

class FU(Piece):

    def __init__(self, init_position, owner, name="FU"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "TO"
        self.move_forward = 1

    def _set_legal_move_komadai(self, banmen):
        positions = np.sum(banmen,axis=(0,1))
        self.legal_move = 1 - positions

        if self.owner==0:
            self.legal_move[:,:self.move_forward] = 0
        else:
            self.legal_move[:,9-self.move_forward:] = 0


        fu_legal = np.sum(banmen[self.turn,12],axis=1)

        self.legal_move *= fu_legal






class KYO(Piece):

    def __init__(self, init_position, owner, name="KY"):
        super().__init__(init_position, owner,name=name)
        self.promote_name = "NY"
        self.move_forward = 1


    def _set_legal_move_board(self,banmen):
        move = self.move_dict[self.name]

        s_positions, g_positions = np.sum(banmen,axis=1)

        if self.name == "NY":
            positions = np.sum(banmen[self.owner], axis=0)

            if self.owner==1:
                move = move[::-1,::-1]
                s_positions_board, g_positions_board = g_positions_board, s_positions_board

            self.legal_move = move[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
            self.legal_move *= 1 - positions


        else:

            if not self.owner:
                s_positions, g_positions = g_positions, s_positions
            s_positions_17, g_positions_17 = np.zeros((2,17,17))
            s_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = s_positions
            g_positions_17[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]] = g_positions
            a = np.zeros((17,17))
            b0 = np.array([True,True,True,True])
            # print(positions)
            for i in range(1,9):
                x = (8,i+8)
                a = b0 * s_positions_17[x]
                b0 = b0 * g_positions_17[x] * a


            self.legal_move = a[9-self.position[0]:18-self.position[0], 9-self.position[1]:18-self.position[1]]
            self.legal_move *= 1-s_positions

    def check_no_promote(self,goal):
        return goal[1]!=(1+8*self.owner)


class KEIMA(Piece):

    def __init__(self, init_position, owner, name="KE"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "NK"
        self.move_forward = 2

    def check_no_promote(self,goal):
        return goal[1] not in [1+8*self.owner, 2+6*self.owner]


class GIN(Piece):

    def __init__(self, init_position, owner, name="GI"):
        super().__init__(init_position, owner, name=name)
        self.promote_name = "NG"

class KIN(Piece):

    def __init__(self, init_position, owner, name="KI"):
        super().__init__(init_position, owner, name=name)

    def check_promote(self, goal):
        return False

class OU(Piece):

    def __init__(self, init_position, owner, name="OU"):
        super().__init__(init_position, owner, name=name)

    def check_promote(self, goal):
        return False

class Board:
    def __init__(self,pieces,players=["P1", "P2"]):
        self.pieces = tuple(pieces)
        self.turn = 1
        self.banmen = np.zeros([2,14,9,9])
        self.komadai = np.zeros([2,14])

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
        self.names = tuple(["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],dtype=str)
        self.piece_class = [OU, HISHA, HIHA, KAKU, KAKU, KIN, GIN, GIN, KEIMA, KEIMA, KYO, KYO, FU, FU]
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

    def get_piece(self, pieces, owner, name, position):
        if position==(0,0):
            return self.get_piece_komadai(pieces, owner, name)
        else:
            return self.get_piece_banmen(pieces, position)

    def get_piece_banmen(self, pieces, position):
        for p in pieces:
            if (p.position==tuple(position)):
                return p
        return None

    def get_piece_komadai(self, pieces, owner, name):
        for p in pieces:
            if (p.position==(0,0)) & (p.owner==owner) & (p.name==name):
                return p
        return None




    def move(self, start, goal, name):

        p = self.get_piece(self.turn, name, start)
        p.position = tuple(goal)

        target = self.get_piece_banmen(goal)
        if target is not None:
            target.captured()

        if name !=p.name:
            p.name = p.promote_name


    def clear(self):
        self.pieces = []
        # self.board_data()

    def board_from_data(self, banmen, komadai):
        self.clear()
        # self.board_data()
        ind_banmen = np.where(banmen).T
        ind_komadai = np.where(komadai).T
        for i in ind_banmen:
            self.pieces.append(self.piece_class[i[1]](init_position=(i[2]+1, i[3]+1), owner=i[0], name=self.names[i[1]]))

        for i in ind_komadai:
            self.pieces.append(self.piece_class[i[1]](init_position=(0,0), owner=i[0], name=self.names[i[1]]))

        self.board_data()


    def move_tmp(self, pieces, start, goal, name):
        p = self.get_piece(pieces, self.turn, name, start)
        p.position = tuple(goal)

        target = self.get_piece_banmen(goal)
        if target is not None:
            target.captured()

        if name !=p.name:
            p.name = p.promote_name
        return pieces

    def move_data_tmp(self, start, goal, name):
        pieces_tmp = copy.deepcopy(self.pieces)
        pieces_tmp = self.move_tmp(pieces_tmp, start, goal, name)

        banmen = np.zeros([2,14,9,9])
        komadai = np.zeros([2,14])

        for p in pieces_tmp:
            if p.position==(0,0):
                komadai[p.owner, self.names_dict[p.name]] =+ 1

            else:
                banmen[p.owner, self.names_dict[p.name], p.position[0]-1, p.position[1]-1] =+ 1

        return banmen, komadai


    def board_data(self):
        self.banmen = np.zeros([2,14,9,9])
        self.komadai = np.zeros([2,14])

        for p in self.pieces:
            if p.position==(0,0):
                komadai[p.owner, self.names_dict[p.name]] =+ 1

            else:
                banmen[p.owner, self.names_dict[p.name], p.position[0]-1, p.position[1]-1] =+ 1

    def read_file(self, filename):
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
        banmen_out = []
        komadai_out = []
        for i,mv in enumerate(move_data):
            # print(mv)
            self.board_data()
            start = [int(mv[0]),int(mv[1])]
            goal = [int(mv[2]),int(mv[3])]
            name = mv[4:6]
            # self.board_data()
            moves = self.get_legal_moves()
            komadai_move = np.zeros([len(moves), 2, 2, 14])
            banmen_move = np.zeros([len(moves), 2, 2, 14, 9, 9])
            komadai_move[:,0,...] = self.komadai
            banmen_move[:,0,...] = self.banmen

            for j,m in enumerate(moves):
                banmen_tmp,komadai_tmp = self.move_data_tmp(start=(int(m[0]),int(m[1])),goal=(int(m[2]),int(m[3])),name=m[4:6])
                komadai_move[j,1,...] = komadai_tmp
                banmen_tmp[j,1,...] = banmen_tmp
            # print(aa.shape)
            if i % 2==1:
                komadai_move = komadai_move[:, :, ::-1]
                banmen_move = banmen_move[:, :, ::-1, :, ::-1, ::-1]
                # aa[:,:,:,:] = aa[:,:,:,::-1]
            # print(moves)
            if not (np.array(moves) == mv[0:6]).any():
                raise ValueError
            good.append((np.array(moves) == mv[0:6]))
            komadai_out.append(komadai_tmp)
            banmen_out.append(banmen_tmp)
            self.move(start, goal, name)

            self.turn = 1- self.turn

        komadai_out = np.concatenate(komadai_out,axis=0)
        banem_out = np.concatenate(banmen_out,axis=0)
        good = np.concatenate(good,axis=0)
        print(filename)
        # print(out.shape)
        return banmen_out, komadai_out, good


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
                p.set_legal_move(self.out_data)
                move_int = np.array(np.where(p.legal_move)).T + 1
                # print(move_int)

                moves =

                for mi in move_int:


        return np.unique(moves)








        # pieceis = [OU(init_posision=[4,8], owner=True),
        #            OU(init_posision=[4,0], owner=False),
        #            ]






    # def fname(arg):
    #     pass
