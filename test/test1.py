import torch
from zviz import Zviz

conv0 = torch.nn.Conv2d(3, 3, 3)
zviz = Zviz({'conv0': conv0})
zviz.setoptimizer(torch.optim.Adam(conv0.parameters()), 'conv0')
data = torch.randn(3, 3, 256, 256)
out = conv0(data)
loss = out.mean()
zviz.backward(loss)
zviz.step('conv0')
zviz.zero_grad('conv0')