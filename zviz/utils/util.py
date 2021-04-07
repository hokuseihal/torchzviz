flatten = lambda x: [z for y in x for z in (flatten(y) if hasattr(y, '__iter__') and not isinstance(y, str) else (y,))]
def flatten_tensor(x):
    ret=[]
    for y in x:
        print(type(y))
        if type(y)==type([]):
            ret.extend(flatten_tensor(y))
        else:
            ret.append(y)
    return ret

if __name__=='__main__':
    import torch
    print(flatten_tensor([[[torch.tensor([1,2,3])]],torch.zeros(3)]))
