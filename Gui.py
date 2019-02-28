"""
    Simulation (GUI) class with an trivial example
"""
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation

# color configs
SOURCE_COLOR = 'green'  # for the assumed non-faulty source node
NON_FAULTY_BEFORE = 'white'  # for non-faulty nodes before committing
NON_FAULTY_AFTER = 'yellow'  # for non-faulty nodes after committing
FAULTY_BEFORE = 'pink'  # for faulty nodes before receiving a value from the source
FAULTY_AFTER = 'red'  # for faulty nodes after receiving a value from the source


class Sim:
    def __init__(self, g=None, d=None):
        if g is None:
            # the below trivial example demonstrates when f = 2,
            # i.e when a non-faulty node see 3 messages with the same value, the node then commits the value
            print("None!!!!!!!!!!!")
            g = nx.DiGraph()
            g.add_nodes_from(range(8))
            g.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)])
            g.add_edges_from([(1, 6), (1, 7)])
            g.add_edges_from([(2, 6), (4, 6), (7, 6)])
            g.add_edges_from([(2, 7), (3, 7), (5, 7)])
            # for n, nbrs in G.adj.items():
            #     for nbr, eattr in nbrs.items():
            #         print(n, nbr, eattr)

            # data structure:
            # a dictionary with integer keys representing round numbers;
            # values are lists of length 2. For each list (value of the dict),
            # the first entry is a list of non-faulty nodes (integers) that commit the message at this round
            # the second entry is a list of faulty nodes that commit the message at this round
            d = {
                0: [[0], []],
                1: [[1, 2, 3], [4, 5]],
                2: [[7], []],
                3: [[6], []]
            }
        self.graph = g
        self.data = d
        self.pos = nx.spring_layout(self.graph)
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
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
                    # for the source node
                    self._draw_nodes(nds[0], SOURCE_COLOR)
                else:
                    # for all non-faulty node
                    self._draw_nodes(nds[0], NON_FAULTY_BEFORE)
                    self._draw_nodes(nds[1], FAULTY_BEFORE)
        # for all other rounds
        else:
            nds = self.data[curr_round]
            self._draw_nodes(nds[0], NON_FAULTY_AFTER)
            self._draw_nodes(nds[1], FAULTY_AFTER)

        self.ax.set_title(f"Round {curr_round}", fontweight="bold")
        self.ax.set_xticks([])
        self.ax.set_yticks([])

    def _simulate(self):
        ani = matplotlib.animation.FuncAnimation(self.fig, self._update, frames=len(self.data.keys()),
                                                 interval=1000, repeat=True)
        plt.show()
