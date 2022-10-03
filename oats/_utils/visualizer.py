from oats._utils import graph_data


class Visualizer:
    def __init__(self, series, labels, train_len):
        self.p = graph_data(series, labels, train_len)
        self.ax1, self.ax2 = self.p.get_axes()

        self.idx = 0

    def add_result(self, anoms: list, model_name: str):
        xmin = 0.125
        ymin = 0.1 - (1 + self.idx) * 0.05
        dx = 0.775
        dy = 0.05

        ax_new = p.add_axes([xmin, ymin, dx, dy], anchor="SW", sharex=self.ax1)
        ax_new.text(10, 0.5, model_name, fontfamily="monospace", va="center")

        ax_new.set_xticklabels([])
        ax_new.set_xticks([])
        ax_new.set_yticklabels([])
        ax_new.set_yticks([])

        ax_new.vlines(anoms, 0, 1, colors="red")
        self.idx += 1
