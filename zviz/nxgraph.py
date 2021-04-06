import networkx as nx
import zviz.utils.project_util as PU
from  zviz.utils.util import  flatten

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
    edgestyle = edgestyledic[phase]
    sucs = flatten([list(G.successors(node)) for node in nodes])
    if sucs==[]:
        return G
    G.remove_nodes_from(getallsuccessorsfromlist(G, nodes, ends, depre=True) + nodes)
    G = nx.compose(G, H)
    if ends!=[]:
        for desuc in ends:
            for h in get(H, root=True):# in fact, loop for once
                backwardids=''
                G.add_edge(desuc, h, style=edgestyle, label=f'grad:{PU.join(" ", backwardids, "")}')
        for end in ends:
            G.add_edge(end, name, style=edgestyle, label=f'grad:')
    for suc in sucs:
        for h in nodes:# in fact, loop for once
            ct = tree.hasthisctree(suc)
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
            G.add_node(ct.id, xlabel=f'step:{PU.join(",",ct.stepids,"")}',
                       label=f'{{{ct.id},{ct.variableid}|grad:{PU.join(" ", ct.getbackgradidslist(), "")}}}',
                       shape='record')
        else:
            if ct.name:
                G.add_node((ct.id), label=f'{PU.takefirst(str(ct.name))},{ct.id}')
            # else:
            #     G.add_node((ct.id), label=f'data')
        if ct.backgradids != []:
            for id, nexts in ct.backgradids:
                for nextid, nextname in nexts:
                    if nextname and ct.id:
                        G.add_edge(str(nextid), ct.id, label=f'grad:{id}', style=edgestyle)


def makegraph(trees, namedinout, phase, savepath, save=False):
    ctrees = trees.ctrees
    G = nx.DiGraph()
    makefromctrees(G, ctrees, phase)


    if save:
        nx.nx_agraph.to_agraph(G).draw(savepath, prog='dot')

    return G
def replacewithmodels(G, namedinout, trees,savepath=None):
    _G=G.copy()
    #set subgraph
    SG=[]
    for key in namedinout:
        name, _datalist, _outputlist, model = namedinout[key]
        datalist=[]
        outputlist=[]
        for idx,xs in enumerate(_outputlist):
            if hex(id(xs[0].grad_fn)) in _G.nodes():
                datalist.append(_datalist[idx])
                outputlist.append(_outputlist[idx])

        dataIdshapes,valdataIdshapes=PU.getidshapedict(datalist)
        outputIdshapes,_=PU.getidshapedict(outputlist)
        for oId in outputIdshapes.keys():
            if oId in _G.nodes():
                pass
        if set(_G.nodes()) & set(outputIdshapes.keys())==set():
            continue
        modelvariableids=[hex(id(p)) for p in model.parameters()]
        modelnodes= getallsuccessorsfromlist(_G, outputIdshapes.keys(), list(dataIdshapes.keys()), depre=True) + list(outputIdshapes.keys())
        steps=trees.getvariablesteps(modelvariableids)
        grads=trees.getvariablebackwards(modelvariableids)
        H=_G.subgraph(modelnodes)
        _G.add_node(name, label=f'{{{name}|grad:{PU.join(" ", grads, "")}}}', xlabel=f'step:{PU.join(" ", steps, "")}', shape='record')
        for valdataid in valdataIdshapes:
            _G.add_node(valdataid, label=f'{valdataIdshapes[valdataid]}', shape='invtriangle')
            _G.add_edge(valdataid, name)
        SG.append([name,H,{**valdataIdshapes,**dataIdshapes},outputIdshapes])
    for name,H,dataIdshapes,outputIdshapes in SG:
        delnodes=[]
        for n in H:
            for idx,nbrs in enumerate([set(_G.predecessors(n)), set(_G.successors(n))]):
                for nbr in set(nbrs) - set(H.nodes()):
                    if idx==1:
                        a,b=name,nbr
                        la,lb=n,nbr
                        outputshape=f'{outputIdshapes[n]}'
                    else:
                        a,b= nbr,name
                        la,lb=nbr,n
                        # outputshape=f'{dataIdshapes[n]}'
                        outputshape = ''
                    grads=_G.edges[la,lb]['label'].split(':')[-1]
                    grads+='\n'+outputshape
                    _G.add_edge(a, b, style='dashed',label=f'grad:{grads}',tail=f'{outputshape}')
            delnodes.append(n)
        _G.remove_nodes_from(delnodes)
    if savepath:
        nx.nx_agraph.to_agraph(_G).draw(savepath, prog='dot')
    return _G

def update(G,params,trees,savepath=None):
    for p in params:
        ct=trees.hasthisctree(hex(id(p)),findvariable=True)
        if G.has_node(ct.id):
            n=G.nodes[ct.id]
            n['xlabel']=f'step:{PU.join(",",ct.stepids,"")}'
            n['label']=f'{{{ct.id},{ct.variableid}|grad:{PU.join(" ", ct.getbackgradidslist(), "")}}}'
    if savepath:
        nx.nx_agraph.to_agraph(G).draw(savepath, prog='dot')
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
