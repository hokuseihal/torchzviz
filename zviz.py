import itertools as I

import nxgraph as nxg
from tree import Tree


# TODO check residual connection
# TODO fix addressing None
class Zviz:
    def __init__(self, nameddic, graphimgpath='tmp.png'):
        self.optim = {}
        self.tree = Tree()
        self.nameddic = nameddic
        self.namedinout = {hex(id(nameddic[k])): [k, [], [], nameddic[k]] for k in
                           nameddic}
        self.graphimgpath = graphimgpath

        def forwardhook(model, data, out):
            mId = hex(id(model))
            # print(mId,self.namedinout)
            if data[0].grad_fn:
                inputid_ori = hex(id(data[0].grad_fn)), data[0].grad_fn
            else:
                inputid_ori = hex(id(data[0])),data[0]
            self.namedinout[mId][1].append([*inputid_ori, data[0].shape])
            if out.grad_fn:
                outid_ori = hex(id(out.grad_fn)), out.grad_fn
            else:
                outid_ori = None, None
            self.namedinout[mId][2].append([*outid_ori, out.shape])
            # print(hex(id(out.grad_fn)))
            # print(out.grad_fn)
            # print(hex(id(out)))

        for name in nameddic:
            model = nameddic[name]
            model.register_forward_hook(forwardhook)

    def checkoptimexist(self):
        assert len(self.optim) != 0, "Do zip.optimizer(your_optimizer)."

    def backward(self, x):
        self.checkoptimexist()
        self.tree.backward(x)
        self.makegraph('backward')
        x.backward()

    def setoptimizer(self, _optim, key='main'):
        self.optim[key] = [_optim, list(I.chain.from_iterable([pg['params'] for pg in _optim.param_groups]))]

    def step(self, key='main'):
        self.checkoptimexist()
        optim, params = self.optim[key]
        self.tree.step(params)
        self.makegraph('step')
        optim.step()

    def zero_grad(self, key='main'):
        self.checkoptimexist()
        optim, params = self.optim[key]
        self.tree.zero_grad(params)
        self.makegraph('zero_grad')
        optim.zero_grad()

    def makegraph(self, phase):
        G = nxg.makegraph(self.tree, self.namedinout, phase, self.graphimgpath, True)


if __name__ == '__main__':
    import test.test5


    print('END')
