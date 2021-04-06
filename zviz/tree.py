BACKWARD = 0
STEP = 1
ZERO_GRAD = 2

from zviz.utils.util import flatten
class Tree:
    class cTree:
        def __init__(self, id, name=None):
            self.id = id
            self.status = 0
            self.stepids = []
            self.backgradids = []
            self.name = name
            self.isvariable = False
            self.variableid = 0
            self.variableshape = 0

        def getbackgradidslist(self):
            return list(set([bids[0] for bids in self.backgradids]))

        def getbackward(self,s_id):
            ret=[]
            for bid,nexts in self.backgradids:
                for nid,_ in nexts:
                    if nid==s_id:
                        ret.append(bid)
            return list(set(ret))

        def backward(self, id, nextid, retain_graph=False):
            # assert not retain_graph, "According my principle, retain_graph is banned."
            if self.isvariable:
                self.checknextstatus(BACKWARD)
            self.backgradids.append([id, nextid])

        def zero_grad(self, ids=None):
            if ids is None:
                self.checknextstatus(ZERO_GRAD)
                ret = self.stepids
                # self.stepids = []
                self.backgradids = [ (None,o) for id,o in self.backgradids]
                return ret
            else:
                # self.stepids = list(set(self.stepids) - ids)
                dels = []
                for idx, (ownid, _) in enumerate(self.backgradids):
                    if ownid in ids:
                        dels.append([ownid, _])
                for i in dels:
                    self.backgradids.remove(i)

        def step(self):
            self.checknextstatus(STEP)
            self.stepids.extend(list(set([t[0] for t in self.backgradids])))

        def checknextstatus(self, key):
            if key == BACKWARD:
                assert self.status != 2, "You need zero_grad() before this backward(loss)."
                self.status = 1
            elif key == STEP:
                assert self.status != 0, "A step() without backward(loss) is meaningless. Maybe bugs?"
                assert self.status != 2, "Double step()s are detected. This operation is inappropriate.You need step() and next iter's backward(loss)"
                self.status = 2
            elif key == ZERO_GRAD:
                # assert self.status != 0, "zero_grad() without backward(loss) and step() is meaningless. Maybe bugs?"
                # assert self.status != 1, "zero_grad() without step() is meaningless. Maybe bugs?"
                self.status = 0

            else:
                assert False, 'A internal error. Please tell me.'

    def __init__(self):
        self.ctrees = []
        self.backid = 0
    #TODO implementation
    def hasthesectrees(self,ids,findvariable):pass
    # def hasthisctree(self, id, findvariable=False, ):
    #     for ctree in self.ctrees:
    #         if (ctree.id if not findvariable else ctree.variableid) == id:
    #             return True, ctree
    #     return False, None

    def hasthisctree(self, id, findvariable=False, ):
        for ctree in self.ctrees:
            if (ctree.id if not findvariable else ctree.variableid) == id:
                return ctree
        return None
    def getAbackwardid(self, params):
        for p in params:
            ct = self.hasthisctree(p, True)
            if ct and ct.stepids != []:
                return ct.stepids
        return []

    def findall(self, variable):
        return [ctree for ctree in self.ctrees if not (ctree.isvariable ^ variable)]
    def getbackwards(self,ids):
        return list(set(flatten([ct.getbackgradidslist() for ct in self.ctrees if ct.id in ids])))

    def getvariablebackwards(self,ids):
        return list(set(flatten([ct.getbackgradidslist() for ct in self.ctrees if ct.variableid in ids])))
    def getsteps(self,ids):
        return list(set(flatten([ct.stepids for ct in self.ctrees if ct.id in ids])))
    def getvariablesteps(self,ids):
        return list(set(flatten([ct.stepids for ct in self.ctrees if ct.variableid in ids])))

    def backward(self, var):
        def follownext(t):
            # print(t, hex(id(t)))
            takenctree = self.hasthisctree(hex(id(t)))
            if not takenctree:
                takenctree = self.cTree(hex(id(t)), name=t)
                if hasattr(t, 'variable'):
                    takenctree.isvariable = True
                    takenctree.variableid = hex(id(t.variable))
                    takenctree.variableshape = t.variable.shape

                self.ctrees.append(takenctree)
            if t is not None and len(t.next_functions) != 0:
                takenctree.backward(self.backid, [(hex(id(_t[0])), _t[0]) for _t in t.next_functions])
                for _t in t.next_functions:
                    follownext(_t[0])
            else:
                takenctree.backward(self.backid, [])

        if var.grad_fn:
            follownext(var.grad_fn)
            self.backid += 1
        else:
            assert False, "This value doesn't have grad_fn. Is computational graph constructed?"

    def step(self, params):
        for p in params:
            takenctree = self.hasthisctree(hex(id(p)), findvariable=True)
            if takenctree:
                takenctree.step()
            else:
                assert False, f'Cannot find {hex(id(p))} in zviz. Maybe backward(loss) is missing.'

    def zero_grad(self, params):
        ids = []
        for p in params:
            takenctree = self.hasthisctree(hex(id(p)), findvariable=True)
            if takenctree:
                ids += takenctree.zero_grad()
            else:
                assert False, f'WHY? CANNOT FIND {hex(id(p))} Maybe you miss backward?'
        # ids = set(ids)
        # for takenctree in self.findall(variable=False):
        #     takenctree.zero_grad(ids)
        # dellist = []
        # for ct in self.ctrees:
        #     if ct.backgradids == [] and ct.status == 0:
        #         dellist.append(ct)
        # for dct in dellist:
        #     self.ctrees.remove(dct)
