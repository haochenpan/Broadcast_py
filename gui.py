"""
    Simulation (GUI) class with an trivial example
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation
from pickle import load
from broadcasting import broadcast_for_gui

# color configs
TRUSTED_BEFORE = 'lightgreen'  # for trusted nodes, e.g. the source node, before committing
TRUSTED_AFTER = 'green'  # for trusted nodes, e.g. the source node, after committing
NON_FAULTY_BEFORE = 'white'  # for non-faulty nodes (but not trusted) before committing
NON_FAULTY_AFTER = 'yellow'  # for non-faulty nodes (but not trusted) after committing
FAULTY_BEFORE = 'pink'  # for faulty nodes before receiving a value from the source
FAULTY_AFTER = 'red'  # for faulty nodes after receiving a value from the source


class Sim:
    def __init__(self, g=None, d=None):
        assert g is not None
        assert d is not None
        self.graph = g
        self.data = d
        self.pos = nx.spring_layout(self.graph)
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self._simulate()

    def _draw_nodes(self, nodelist, node_color):
        nx.draw_networkx_nodes(self.graph, pos=self.pos, ax=self.ax,
                               nodelist=nodelist, node_color=node_color,
                               edgecolors="black")

    def _update(self, curr_round):
        # initialization work in the first round
        if curr_round == 0:
            nx.draw_networkx_edges(self.graph, pos=self.pos, ax=self.ax, edge_color="gray")
            nx.draw_networkx_labels(self.graph, self.pos)
            for rd, nds in self.data.items():
                if rd == 0:
                    self._draw_nodes(nds[2], TRUSTED_AFTER)  # b/c the source has committed the value
                else:
                    self._draw_nodes(nds[0], NON_FAULTY_BEFORE)
                    self._draw_nodes(nds[1], FAULTY_BEFORE)
                    self._draw_nodes(nds[2], TRUSTED_BEFORE)  # b/c the source has committed the value
        # for all other rounds
        else:
            nds = self.data[curr_round]
            self._draw_nodes(nds[0], NON_FAULTY_AFTER)
            self._draw_nodes(nds[1], FAULTY_AFTER)
            self._draw_nodes(nds[2], TRUSTED_AFTER)

        self.ax.set_title(f"Round {curr_round}", fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _simulate(self):
        ani = matplotlib.animation.FuncAnimation(self.fig, self._update, frames=len(self.data.keys()),
                                                 interval=1000, repeat=True)
        plt.show()


if __name__ == '__main__':
    graph_path = '/Users/haochen/Desktop/Broadcast_py/subgraphs/geo_200/geo_200_0.2_1551507763.pi'
    G = load(open(graph_path, "rb"))
    trust_nodes = {0, 20, 40, 60, 80}
    faulty_nodes = {15, 35, 90, 120}
    commits_count, run_dict = broadcast_for_gui(G, faulty_nodes, trust_nodes)
    Sim(g=G, d=run_dict)
