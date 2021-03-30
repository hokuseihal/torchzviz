import itertools as I

import nxgraph as nxg
from tree import Tree


class Zviz:
    def __init__(self, nameddic, graphimgpath='tmp.png'):
        self.optim = {}
        self.tree = Tree()
        self.nameddic = nameddic
        self.namedinout = {hex(id(nameddic[k])): [k, [None, None, None], [None, None, None], nameddic[k]] for k in
                           nameddic}
        self.graphimgpath = graphimgpath

        def forwardhook(model, data, out):
            mId = hex(id(model))
            # print(mId,self.namedinout)
            if data[0].grad_fn:
                self.namedinout[mId][1][0] = hex(id(data[0].grad_fn))
                self.namedinout[mId][1][1] = data[0].grad_fn
            self.namedinout[mId][1][2] = data[0].shape
            if out.grad_fn:
                self.namedinout[mId][2][0] = hex(id(out.grad_fn))
                self.namedinout[mId][2][1] = out.grad_fn
            self.namedinout[mId][2][2] = out.shape
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
    import torch
    from torchvision.models import resnet18

    conv0 = torch.nn.Conv2d(3, 3, 3)
    # conv1 = torch.nn.Conv2d(3, 3, 3)
    # conv2 = resnet18()
    # conv2=torch.nn.Sequential(torch.nn.Conv2d(3,3,3),torch.nn.Conv2d(3,3,3))
    # model = torch.nn.Sequential(conv0, conv1)
    zviz = Zviz({'conv0': conv0 })
    optim = torch.optim.Adam(conv0.parameters())
    # optim2 = torch.optim.Adam(conv2.parameters())
    zviz.setoptimizer(optim, 'model')
    # zviz.setoptimizer(optim2, '2')
    data = torch.randn(3, 3, 256, 256)
    data2 = torch.randn(3, 3, 256, 256)
    out = conv0(data)
    out2 = conv0(data2)
    loss = out.mean()
    loss2 = out2.mean()
    zviz.backward(loss)
    zviz.backward(loss2)
    # zviz.step('model')
    # zviz.step('2')

    # zviz.zero_grad('model')
    # zviz.zero_grad('2')
    print('END')
