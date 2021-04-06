def join(s=" ", li=[], empty_s=''):
    li=[l for l in li if l !=None]
    if li == []:
        return empty_s
    elif len(li) == 1:
        return li[0]
    else:
        return s.join(str(l) for l in li)


def takefirst(s):
    if s[0] in ['<', '>', ':']: s = s[1:]
    return str(s).split()[0]


def printgradfn(x):
    def printnext(t):
        print(t, hex(id(t)))
        if hasattr(t, 'variable'):
            print('V', hex(id(t.variable)), t.variable.shape)

        if t is not None and len(t.next_functions) != 0:
            print(t.next_functions)
            for _t in t.next_functions:
                printnext(_t[0])

    if x.grad_fn:
        printnext(x.grad_fn)
        print('')

def getidshapedict(xslist):
    Idshapes={}
    Idshapes_nograd={}
    for i in range(len(xslist)):
        for numx in range(len(xslist[i])):
            x=xslist[i][numx]
            if x.grad_fn:
                Idshapes[hex(id(x.grad_fn))]=xslist[i][numx].shape
            else:
                Idshapes_nograd[hex(id(x))] = xslist[i][numx].shape
    return Idshapes,Idshapes_nograd