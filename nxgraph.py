import networkx as nx

import utils.project_util as PU
from utils.util import flatten


def getallsuccessors(G, node, sucs, end=None, depre=False):
    fn = G.predecessors if depre else G.successors
    for suc in fn(node):
        # print(f'{node} ->{suc}')
        if end is None or suc != end:
            getallsuccessors(G, suc, sucs, end, depre=depre)
            sucs.append(suc)
    return sucs


def replacefrom(G, node, H, end=None, phase='forward', datashape=None, outputshape=None, backwardids=[]):
    edgestyle = edgestyledic[phase]
    if end: predecessors = list(G.predecessors(end))
    sucs = G.successors(node)
    G.remove_nodes_from(getallsuccessors(G, node, [], end, depre=True) + [node])
    G = nx.compose(G, H)
    if end:
        for desuc in predecessors:
            for h in get(H, root=True):
                G.add_edge(desuc, h, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')
        G.add_edge(end, node, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')
    for suc in sucs:
        for h in get(H, root=False):
            G.add_edge(h, suc, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')

    return G


def get(G, node=None, root=False):
    fn = G.predecessors if root else G.successors
    if node is None: node = next(iter(G.nodes))
    sucs = list(fn(node))
    if sucs == []:
        return [node]
    else:
        return flatten([get(G, suc, root) for suc in sucs])


edgestyledic = {'forward': 'solid', 'backward': 'dashed', 'step': 'dashed', 'zero_grad': 'dotted'}


def makefromctrees(G, ctrees, phase):
    edgestyle = edgestyledic[phase]
    for ct in ctrees:
        if ct.isvariable:
            G.add_node(ct.id, xlabel=f'step:{ct.stepids}',
                       label=f'{{{ct.variableshape},{ct.variableid}|grad:{PU.join(" ", ct.getbackgradidslist(), "")}}}',
                       shape='record')
        else:
            if ct.name != 'None':
                G.add_node((ct.id), label=f'{PU.takefirst(str(ct.name))},{ct.id}')
            else:
                G.add_node((ct.id), label=f'data')
        if ct.backgradids != []:
            for id, nexts in ct.backgradids:
                for nextid, nextname in nexts:
                    G.add_edge(str(nextid), ct.id, label=f'grad:{id}', style=edgestyle)


def makegraph(trees, namedinout, phase, savepath, save=False):
    ctrees = trees.ctrees
    G = nx.DiGraph()
    makefromctrees(G, ctrees, phase)

    for key in namedinout:
        name, data, output, model = namedinout[key]
        H = nx.DiGraph()
        dataId, _, datashape = data
        outputId, _, outputshape = output
        if G.has_node(outputId):
            ct = trees.hasthisctree(outputId)[1]
            stepids = trees.getAbackwardid([hex(id(p)) for p in model.parameters()])
            backwardids = ct.getbackgradidslist()
            H.add_node(str(outputId), xlabel=f'step:{PU.join(" ", stepids, "")}',
                       label=f'{{name:{name}|grad:{PU.join(" ", backwardids, "")}|input:{datashape}|output:{outputshape}}}',
                       shape='record')
            G = replacefrom(G, outputId, H, dataId, phase, datashape, outputshape, backwardids)
    if save:
        nx.nx_agraph.to_agraph(G).draw(savepath, prog='dot')

    # pos=nx.spring_layout(G)
    # nx.draw_networkx(G,pos)
    # import matplotlib.pyplot as plt
    # plt.show()
    return G


if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_node('a', color='black', label='{0x99ec989|1 2 3}', shape='record', xlabel='1 2 3')
    G.add_edge('b', 'a', color='black:invis:black', width='20', label='')
    G.add_edge('a', 'c')
    G.add_edge('a', 'd')
    G.add_edge('c', 'e')
    G.add_edge('e', 'f')
    H = nx.DiGraph()
    H.add_edge('h', 'i', label='hi')
    H.add_edge('i', 'j', label='ij')
    G = replacefrom(G, 'a', H, 'e')
    nx.nx_agraph.to_agraph(G).draw('file.png', prog='fdp')
