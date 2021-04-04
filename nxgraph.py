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

def getallsuccessorsfromlist(G,nodes,ends=[],depre=False):
    ret=[]
    if ends!=[]:
        for node in nodes:
            for end in set(ends):
                ret.extend(getallsuccessors(G,node,[],end,depre))
    else:
        for node in nodes:
            ret.extend(getallsuccessors(G,node,[],depre=depre))
    return list(set(ret))

def replacefrom(G, nodes, H, ends=[], phase='forward', datashape=None, outputshape=None,name="",tree=None,endbackids=[]):
    # assert len(ends)==len(endbackids)
    edgestyle = edgestyledic[phase]
    # if ends!=[]: predecessors = flatten([list(G.predecessors(end)) for end in ends]) !!!
    sucs = flatten([list(G.successors(node)) for node in nodes])
    if sucs==[]:
        return G
    G.remove_nodes_from(getallsuccessorsfromlist(G, nodes, ends, depre=True) + nodes)
    G = nx.compose(G, H)
    if ends!=[]:
        for desuc in ends:
            for h in get(H, root=True):# in fact, loop for once
                # flag,ct=tree.hasthisctree(desuc)###
                # backwardids=ct.getbackward(h)###
                backwardids=''
                G.add_edge(desuc, h, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')
        for end in ends:
            G.add_edge(end, name, style=edgestyle, label=f'grad:')
    for suc in sucs:
        for h in nodes:# in fact, loop for once
            flag, ct = tree.hasthisctree(suc)
            backwardids = ct.getbackward(h)
            if backwardids!=[]:
                G.add_edge(name, suc, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')

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
                       label=f'{{{ct.id},{ct.variableid}|grad:{PU.join(" ", ct.getbackgradidslist(), "")}}}',
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

    nx.nx_agraph.to_agraph(G).draw(savepath, prog='dot')
    namelist=[name for name,_,_,_ in namedinout.values()]
    #set subgraph -> for ..
    #replace models' subgraph with a node -> for ..

    for key in namedinout:
        name, datalist, outputlist, model = namedinout[key]
        H = nx.DiGraph()
        dataIds=[]
        datashapes=[]
        valdataIds=[]
        valdatashapes=[]
        databackgradIds=[]
        for dataid , _,datashape in datalist:
            flag,ct=trees.hasthisctree(dataid)
            if flag or dataid in namelist:
                dataIds.append(dataid)
                datashapes.append(datashape)

            else:
                valdataIds.append(dataid)
                valdatashapes.append(datashape)
        outputIds=[]
        outputshapes=[]
        backwardids=[]
        for outputid ,_,outputshape in outputlist:
            flag,ct=trees.hasthisctree(outputid)
            if flag:
                outputIds.append(outputid)
                outputshapes.append(outputshape)
        for p in model.parameters():
            flag,ct=trees.hasthisctree(hex(id(p)),findvariable=True)
            if flag:
                backwardids.extend(ct.getbackgradidslist())
        backwardids=list(set(backwardids))
        stepids = trees.getAbackwardid([hex(id(p)) for p in model.parameters()])
        H.add_node(name, xlabel=f'step:{PU.join(" ", stepids, "")}',
                   label=f'{{name:{name}|grad:{PU.join(" ", backwardids, "")}|input:{datashape}|output:{outputshape}}}',
                   shape='record')
        for valid ,valshape in zip(valdataIds,valdatashapes):
            H.add_node(valid,label=f'{valshape},{valid}',shape='invtriangle')
            H.add_edge(valid,name)

        nx.nx_agraph.to_agraph(G).draw(savepath, prog='dot')
        G = replacefrom(G, outputIds, H, dataIds, phase, datashapes, outputshapes,name,trees)

        for exckey in namedinout.keys()-key:
            for _outputid in outputIds:
                _name, _datalist, _outputlist, _model = namedinout[exckey]
                _datalist=[[_dataid,_origin,_datashape] if _dataid!=_outputid else [name,_origin,_datashape] for _dataid ,_origin,_datashape in _datalist]
                namedinout[exckey]=_name, _datalist, _outputlist, _model

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
