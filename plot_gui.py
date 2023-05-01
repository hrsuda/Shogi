import matplotlib.pyplot as plt
import numpy as np
import pickle
import init_position
import matplotlib.patches as patches
from matplotlib import rcParams
from shogi_3 import *

piece_text = ['王', '飛', '竜', '角', '馬', '金', '銀', '成銀', '桂', '成桂', '香', '成香', '歩', 'と']
names_dict = dict(zip(
["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
['王', '飛', '竜', '角', '馬', '金', '銀', '成銀', '桂', '成桂', '香', '成香', '歩', 'と']
))


class plot_gui:

	def __init__(self, board=None):
		if board is None:
			with open("init_posision.pkl", "rb") as f:
            	pieces = pickle.load(f)
        	board = Board(pieces)
        self.board = board
        self.fig, self.ax = plt.subplots()
        self.active = False
        self.leagal_move = None


        self._set_fig()



    def _set_fig(self):
	    if language =="japanese":
	        rcParams['font.family'] = 'sans-serif'
	        rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    	x = np.arange(0,10) + 0.5
	    y = np.arange(0,10) + 0.5
	    


	    self.ax.invert_xaxis()
	    self.ax.invert_yaxis()

	    self.ax.set_xlim(-0.5,10.5)
	    self.ax.set_ylim(-0.5,10.5)


	    self.ax.hlines(y, x.min(), x.max())
	    self.ax.vlines(x, y.min(), y.max())
	    self.ax.axis("off")
	    self.komadai_count = 0
	    self.komadai = [[],[]]
	    self.banmen = np.zeros((9,9))



	def _plot_postions(self):
	    self.komadai = [[],[]]
	    self.komadai_count = 0
		
		for p in self.board.pieces:
        # print(p.name)
	        position = np.array(p.position)
	        # print(position)
	        if position[0] == 0:
	            xx = 11 * p.owner
	            yy = 1 + 0.5 * self.komadai_count[p.owner]
	            self.ax.text(xx, yy, names_dict[p.name])
	            self.komadai_count[p.owner] += 1
	            self.komadai[p.owner].append(p)


	        else:
	            xx = x[position[0]-1] + 0.8
	            yy = y[position[1]-1] + 0.5
	            self.ax.text(xx, yy, names_dict[p.name], rotation=180*(1-p.owner))
	            self.banmen[position-1] = p
	    self.ax.set_aspect(1)




	def click_position(self, event):
	    print('{} click: button={}, x={}, y={}, xdata={}, ydata={}'.format(
	        'double' if event.dblclick else 'single', event.button,
	         event.x, event.y, event.xdata, event.ydata,
	    ))

	    x = set_x(event.xdata)
	    y = set_x(event.ydata)
	    if (x is None) or (y is None):
	    	return None
	    else:
	    	return (x,y)


	def set_x(self, x):
		if x is None:
			return None

		else:
			ind_min = np.abs(np.arange(0,12) - x).argmin()

			return np.arange(0,12)[ind_min]
		
	




	def action(self, event):

		position = self.click_position(event)

		if self.active:
			if position is None:
				self.active = False
			elif not self.legal_move[position]:
				self.active = False

			else:

				self.move(start=self.start, goal=position, name=self.name)
				self.active = False

		else:
			if not position is None:
				if position[0] == (self.Board.turn * 11):
					position[0] = 0
				self.active = True
				self.position = position
				if position[0] == 0:
					self.piece = self.komadai[self.turn][self.position[1]]

				else:
					self.piece = self.banmen[self.position]


	def set_legal_move(self, piece):
		self.piece.set_legal_move(self.)
		self.legal_move = 




	def move(self, goal):
		
		

		self.Board.move()





def __main__():
	fig, ax = plt.subplots()

