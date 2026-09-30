#!/usr/bin/env python
# update june 16 2025, sorting the last elements of M see getTips() and near return m,M
import sys
import tree
from tree import Tree, Node
import numpy as np

class mmtree(Tree):
    i = 0
    def __init__(self,root=Node()):        
        super().__init__(root)
        #print(Tree.__module__)
        #print(self.root)
        
    def printTiplabels(self,p):
        if not(tree.istip(p)):
            self.printTiplabels(p.left)
            self.printTiplabels(p.right)
        else:
            print (p.name)
        
            
    def getTips(self,p, tips, tipslength):
        if not(tree.istip(p)):
            self.getTips(p.left, tips, tipslength)
            self.getTips(p.right, tips, tipslength)
        else:
            tips.append(p)
            tipslength.append([p.name, p.blength])


    def findPath(self, root, path, n):
            if root is None:
                return False

            path.append(root)
            if root.name == n :
                return True

            if ((root.left != -1 and self.findPath( root.left, path, n)) or
                    (root.right!= -1 and self.findPath( root.right, path, n))):
                return True

            path.pop()
            return False

    def Root_MRCA(self, root, n1, n2):
        path1 = []
        path2 = []
        tn1 = n1
        while(tn1 != root):
            path1.append(tn1)
            tn1 = tn1.ancestor
        path1.append(root)
        #print("@",path1)
        tn2 = n2
        while(tn2 != root):
            path2.append(tn2)
            tn2 = tn2.ancestor
        path2.append(root)
        #print("@", path2)
        rpath1 = path1[::-1] 
        rpath2 = path2[::-1]
        first_common = rpath1[0] # the root
        slen = 0.0
        snum = 0
        # start at first non-root element in reverse path
        for i in range(1,min(len(rpath1), len(rpath2))):
            if rpath1[i] == rpath2[i]:
                first_common = rpath1[i]
                slen += first_common.blength
                snum += 1
            else:
                # Stop at the first point of divergence
                break
        return snum,slen, first_common.age

    def PairTips(self,p, tips, PairList):
        newtips = tips.copy()
        newtips.sort(key=lambda x: x.name)
        for i in range(len(newtips)):
            for j in range(i+1, len(newtips)):
                PairList.append([newtips[i], newtips[j]])
        PairList.sort(key=lambda x: (x[0].name, x[1].name))
        return PairList


    def MetricsVectors(self, newick,tips,tipslength,PairList):
        m=[]
        M=[]
        a=[]
        inter=[]
        #self.myread(newick,self.root)          
        self.getTips(self.root, tips, tipslength)
        tips.sort(key=lambda tip: tip.name)
        #[print(ti.name,ti.ancestor) for ti in tips]
        #sys.exit()
        #print(newick)
        self.PairTips(self.root, tips, PairList)
        for pair in PairList:
            pop1 = int(pair[0].name.split('_')[0])
            pop2 = int(pair[1].name.split('_')[0])
            #interelement = pop2 - pop1
            interelement = (pop1,pop2)
            #pair[0].debugprint()
            numEdgePath, LengthPath, age  = self.Root_MRCA(self.root, pair[0], pair[1])
            a.append(age)
            inter.append(interelement)
            m.append(numEdgePath)
            M.append(LengthPath)
        #print(newick)
        ptips=[1] * len(tips)
        #print('ptips=',ptips)
        m=m+ptips
        tipslength.sort() # sort according to name
        M=M+[ti[1] for ti in tipslength]
        return (m, M, a, inter)
    

if __name__ == "__main__":
    m=[]
    M=[]
    age=[]
    inter=[]
    for i in range(2):
        PairList=[]
        tips=[]
        tipslength=[]
        mmtree = mmtree()
        tips=[]
        a = MetricsVectors(tree, tips, tipslength, PairList)
        m.append(a[0])
        M.append(a[1])
        age.append(a[2])
        inter.append(a[3])
    #print('m=',m ,'\nM=',M, f"{age=}, {inter=}")
