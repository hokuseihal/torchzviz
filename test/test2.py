import torch

from zviz import Zviz

conv0 = torch.nn.Conv2d(3, 3, 3)
conv1 = torch.nn.Conv2d(3, 3, 3)
criteorion0 = torch.nn.MSELoss()
criteorion1 = torch.nn.MSELoss()
zviz = Zviz({'conv0': conv0, 'conv1': conv1, 'mseloss0': criteorion0, 'mseloss1': criteorion1})
optim = torch.optim.Adam(torch.nn.Sequential(conv0, conv1).parameters())
zviz.setoptimizer(optim, 'model')
data = torch.randn(3, 3, 256, 256)
data2 = torch.randn(3, 3, 256, 256)
out = conv0(data)
out2 = conv1(data2)
loss = criteorion0(out, torch.zeros(1))
loss2 = criteorion1(out2, torch.zeros(1))
zviz.backward(loss)
zviz.backward(loss2)
zviz.step('model')
zviz.zero_grad('model')
