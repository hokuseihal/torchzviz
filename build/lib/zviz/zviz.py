import itertools as I

import zviz.nxgraph as nxg
from zviz.tree import Tree
import os

class Zviz:
    def __init__(self, nameddic, graphdir='zviz',_optim=None):
        self.optim = {}
        self.tree = Tree()
        self.nameddic = nameddic
        self.namedinout = {hex(id(nameddic[k])): [k, [], [], nameddic[k]] for k in
                           nameddic}
        self.graphdir = graphdir
        self.graphid=0
        os.makedirs(self.graphdir,exist_ok=True)
        if _optim:
            self.optim=_optim

        def forwardhook(model, data, out):
            mId = hex(id(model))
            # print(mId,self.namedinout)
            self.namedinout[mId][1].append(data)
            self.namedinout[mId][2].append([out])
            # print(hex(id(out.grad_fn)))
            # print(out.grad_fn)
            # print(hex(id(out)))

        for name in nameddic:
            model = nameddic[name]
            model.register_forward_hook(forwardhook)
    def addparams(self,dic):
        self.nameddic={**self.nameddic,**dic}

    def checkoptimexist(self):
        assert len(self.optim) != 0, "Do zip.optimizer(your_optimizer)."

    def backward(self, x):
        self.checkoptimexist()
        self.tree.backward(x)
        self.graphimgpath=f'{self.graphdir}/{self.graphid}_backward.png'
        self.graphid+=1
        self.makegraph('backward')
        x.backward()

    def setoptimizer(self, _optim, key='main'):
        self.optim[key] = [_optim, list(I.chain.from_iterable([pg['params'] for pg in _optim.param_groups]))]
    def step(self, key='main'):
        self.checkoptimexist()
        optim, params = self.optim[key]
        self.tree.step(params)

        self.graphimgpath=f'{self.graphdir}/{self.graphid}_step_{key}.png'
        self.graphid+=1

        self.update_graph(params)
        self.replacewithmodels()
        optim.step()

    def zero_grad(self, key='main'):
        self.checkoptimexist()
        optim, params = self.optim[key]
        self.tree.zero_grad(params)

        self.graphimgpath=f'{self.graphdir}/{self.graphid}_zerograd_{key}.png'
        self.graphid+=1

        self.update_graph(params)
        self.replacewithmodels()
        optim.zero_grad()

    def makegraph(self, phase):
        #TODO update from G which already exists
        self.G = nxg.makegraph(self.tree, self.namedinout, phase, self.graphimgpath, True)
        self.replacewithmodels()

    def update_graph(self, params):
        nxg.update(self.G,params,self.tree,savepath=self.graphimgpath)

    def replacewithmodels(self):
        nxg.replacewithmodels(self.G,self.namedinout,self.tree,self.graphimgpath)

    def clear(self):
        self.__init__(nameddic=self.nameddic,_optim=self.optim)


if __name__ == '__main__':
    import test.test4


    print('END')
