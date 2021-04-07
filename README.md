Zviz
====
A package visualizing PyTorch's traced graph and progress of the nn.Module (backward,step, and zero_grad) of your pytorch.
This repository is inspired by [PyTorchViz](https://raw.githubusercontent.com/szagoruyko/pytorchviz/).


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18K71_mChwZYPl3nN9Q8BezJeN9wgi3b6?usp=sharing)
[![Python](https://img.shields.io/pypi/pyversions/zviz.svg?style=plastic)](https://badge.fury.io/py/zviz)
[![PyPI](https://badge.fury.io/py/zviz.svg)](https://badge.fury.io/py/zviz)
## TODO
- [x] show traced graph and the progress(backward, step, and zero_grad)
- [x] show graph simply with named model
- [x] multiple input, output, backward, step, and zero_grad
- [ ] rich graphics with some animation libraly
## Dependency
- graphviz
  (e.g. ```sudo apt install graphviz libgraphviz-dev```)
- python>=3.7
## Installation
```
pip install zviz
```

## Usage
I prepared some examples on the [colab](https://colab.research.google.com/drive/18K71_mChwZYPl3nN9Q8BezJeN9wgi3b6?usp=sharing).

```
import torch

from zviz import Zviz

conv0 = torch.nn.Conv2d(3, 3, 3)
conv1 = torch.nn.Conv2d(3, 3, 3)

criteorion0 = torch.nn.MSELoss()
criteorion1 = torch.nn.MSELoss()

# set your models and losses with its name
zviz = Zviz({'conv0': conv0, 'conv1': conv1, 'mseloss0': criteorion0, 'mseloss1': criteorion1}, graphdir='zvizimgs')
optim = torch.optim.Adam(conv0.parameters())
optim2 = torch.optim.Adam(conv1.parameters())

# set your optimizer with a key
zviz.setoptimizer(optim, 'conv0')
zviz.setoptimizer(optim2, 'conv1')

for one_batch in range(5):
    data = torch.randn(3, 3, 256, 256)
    data2 = torch.randn(3, 3, 256, 256)
    out = conv0(data)
    out2 = conv1(out)
    loss2 = criteorion0(out, torch.zeros(1)) + criteorion1(out2, torch.zeros(1))

    # use backward, step, and zero_grad of zviz
    zviz.backward(loss2)
    zviz.step('conv0')
    zviz.step('conv1')
    zviz.zero_grad('conv0')
    zviz.zero_grad('conv1')

    # Since zviz holds entire torch graph, you need to call clear() at the end.
    zviz.clear()
    # I assume you only want to create the graph image once, so you can disable zviz and use it as a normal optimizer or loss variant..
    zviz.disable_forever()

```
![image](https://github.com/hokuseihal/torchzviz/blob/master/example.gif)
