import matplotlib.pyplot as plt
import numpy as np
import pickle
import init_position
import matplotlib.patches as patches
from matplotlib import rcParams

piece_text = ['王', '飛', '竜', '角', '馬', '金', '銀', '成銀', '桂', '成桂', '香', '成香', '歩', 'と']
names_dict = dict(zip(
["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"],
['王', '飛', '竜', '角', '馬', '金', '銀', '成銀', '桂', '成桂', '香', '成香', '歩', 'と']
))

def plot_board(pieces, language="japanese"):
    if language =="japanese":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    x = np.arange(0,10) + 0.5
    y = np.arange(0,10) + 0.5
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())
    komadai_count =[0,0]
    for p in pieces:
        # print(p.name)
        position = np.array(p.position)
        # print(position)
        if position[0] == 0:
            xx = 13 * (1-p.owner) - 1
            yy = 1 + 0.5 * komadai_count[1-p.owner]
            ax.text(xx, yy, names_dict[p.name])
            komadai_count[1-p.owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, names_dict[p.name], rotation=180*(1-p.owner))

    ax.set_aspect(1)
    return ax

def plot_board_alpha(pieces, language="japanese"):
    if language =="japanese":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    x = np.arange(0,10) + 0.5
    y = np.arange(0,10) + 0.5
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())
    komadai_count =[0,0]
    for p in pieces:
        # print(p.name)
        position = np.array(p.position)
        # print(position)
        if position[0] == 0:
            xx = 13 * p.owner - 1
            yy = 1 + 0.5 * komadai_count[p.owner]
            ax.text(xx, yy, names_dict[p.name])
            komadai_count[p.owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, names_dict[p.name], rotation=180*(p.owner))

    ax.set_aspect(1)
    return ax

def plot_board_from_data(data, language="japanese"):
    if language =="japanese":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    names = np.array(piece_text)
    x = np.arange(0,10) + 0.5
    y = np.arange(0,10) + 0.5
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())

    ind = np.array(np.where(data)).T

    komadai_count =[0,0]

    for i in ind:
        # print(i)
        owner = i[1]
        name = names[i[0]]
        # print(p.name)
        position = i[2:4]
        # print(position)
        if position[0] == 0:
            num = data1[tuple(i)]

            for n in range(int(num)):
                xx = 13 * (owner) - 1
                yy = 1 + 0.5 * komadai_count[owner]
                ax.text(xx, yy, name)
                komadai_count[owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, name, rotation=180*(owner),)

        ax.set_aspect(1)
    return ax


def plot_move(data1,data2, language="japanese"):
    if language =="japanese":
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']

    # names = np.array(["OU", "HI", "RY", "KA", "UM", "KI", "GI", "NG", "KE", "NK", "KY", "NY", "FU", "TO"])
    names = np.array(piece_text)
    x = np.arange(0,10) + 0.5
    y = np.arange(0,10) + 0.5
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())
    data_diff = data2 - data1

    ind = np.array(np.where(data1)).T
    ind_diff = np.array(np.where(data_diff)).T
    komadai_count =[0,0]

    for i in ind:
        # print(i)
        owner = i[1]
        name = names[i[0]]
        # print(p.name)
        position = i[2:4]
        # print(position)
        if position[0] == 0:
            num = data1[tuple(i)]

            for n in range(int(num)):
                xx = 13 * (owner) - 1
                yy = 1 + 0.5 * komadai_count[owner]
                ax.text(xx, yy, name)
                komadai_count[owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, name, rotation=180*(owner),)

    for ii in ind_diff:
        print(ii)

        # if ii[1]==0:
        if ii[-1]==0:
            t = names[ii[0]]

        else:
            r = patches.Rectangle(xy=(ii[2]-0.5, ii[3]-0.5), width=1, height=1,alpha=0.3,fc='r')
            # ax.text(ii[2]-0.2, ii[3], names[ii[0]], rotation=180*(ii[1]),fontname="MS Gothic",alpha=0.3)

            ax.add_patch(r)


    ax.set_aspect(1)
    if len(ind_diff)>4:
        raise ValueError

    return ax



def main():
    init_position.main()


    with open("init_posision.pkl", "rb") as f:
        pieces = pickle.load(f)
    pieces[10].position = (0,0)
    pieces[15].position = (0,0)
    pieces[24].position = (0,0)
    pieces[9].position = (0,0)


    ax = plot_boad(pieces)

    plt.show()





if __name__ == "__main__":
    main()
