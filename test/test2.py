import torch
from zviz import Zviz

conv0 = torch.nn.Conv2d(3, 3, 3)
conv1 = torch.nn.Conv2d(3, 3, 3)
zviz = Zviz({'conv0': conv0, 'conv1': conv1})
optim = torch.optim.Adam(torch.nn.Sequential(conv0, conv1).parameters())
zviz.setoptimizer(optim, 'model')
data = torch.randn(3, 3, 256, 256)
data2 = torch.randn(3, 3, 256, 256)
out = conv0(data)
out2 = conv1(data2)
loss = out.mean()
loss2 = out2.mean()
zviz.backward(loss)
zviz.backward(loss2)
zviz.step('model')
# zviz.zero_grad('model')