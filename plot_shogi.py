import matplotlib.pyplot as plt
import numpy as np


def plot_boad(pieces):
    x = np.arange(0,10)
    y = np.arange(0,10)
    ax = plt.subplot(111)
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.hlines(y,x.min(),x.max())
    ax.vlines(x, y.min(), y.max())
    komadai_count =[0,0]
    for p in pieces:
        position = p.position
        if position==[0,0]:
            xx = 11 * p.owner - 1
            yy = 1 + 0.5 * komadai_count[~p.owner]
            ax.text(xx, yy, p.name)
            komadai_count[~p.owner] += 1
        else:
            xx = x[position[0]-1] + 0.8
            yy = y[position[1]-1] + 0.5
            ax.text(xx, yy, p.name, rotation=108*p.owner)

        ax.set_aspect(1)
        return ax
