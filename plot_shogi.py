import matplotlib.pyplot as plt
import numpy as np
import pickle
import init_position


def plot_board(pieces):
    x = np.arange(0,10) + 0.5
    y = np.arange(0,10) + 0.5
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())
    komadai_count =[0,0]
    for p in pieces:
        print(p.name)
        position = p.position
        if position == (0,0):
            xx = 13 * (1-p.owner) - 1
            yy = 1 + 0.5 * komadai_count[1-p.owner]
            ax.text(xx, yy, p.name)
            komadai_count[1-p.owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, p.name, rotation=180*(1-p.owner))

        ax.set_aspect(1)
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
