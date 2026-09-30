#!/usr/bin/env python3

# node and tree class
# class Node defines:
#     -Node (see __init__)
#     -tip          : defines tip node and sets label
#     -interior     : defines interior node and sets connections left and right
#     -myprint      : prints name and branchlength associated with a node
#     -debugprint   : prints the content of a node
# class Tree defines:
#     -Tree (see __init__) : defines root, basefrequencies and JukesCantor rate matrix
#     -myprint      : prints the tree in NEWICK format recursively
#     -treeprint    : calls myprint and addes 0.0 and prints to file;
#     -myread       : reads a NEWICK string and creates a tree
#     -printTiplabels: prints tip labels
#     -insertSequence: uses a list of labels and sequences to match the
#                      labels with the tip labels and insert the conditioinal
#                      likelihoods into the tips according to the data
#                      in the sequences list
#     -condLikelihood: internal function to calculate the conditionals in all nodes
#                      (downpass)
#     -likelihood    : calculates the log likelihood of the whole tree
#                      (uses Tree.condLikelihood)
#     -optimizeNR    : optimize branch length (likelihood) using Newton-Raphson {works on some trees}
#     -optimize      : use Nelder Mead optimization
#     -paupoptimize  : hack to faster optimize using paup [needs to be in the path as paup]
#     -setParameters : sets the mutation rate matrix and basefrequencies
#     -printLnL      : prints log likelihood of tree
#     -getLnL        : returns log likelihood
# utility functions:
# - getName          : extracts the lable from the newickstring
# - istip            : returns true if node is a tip


import io
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent = file.parent
sys.path.append(str(file.parent))

import os
import numpy as np
from numpy.random import default_rng
#from scipy.optimize import minimize
#import likelihood as like
#import phylip as phy
#from scipy.optimize import linprog
#import optimize as opt

DEBUG=False


def getName(s):
    name = ""
    for j in range(len(s)):
        if((s[j]==')') | (s[j]=='(') | (s[j]==':') | (s[j]==' ') | (s[j]==',')):
            return j,name
        else:
            name= name + s[j]
    return j,name


def istip(p, ignore_root=False):
    if ignore_root:
        return ((p.left == -1) & (p.right == -1))
    else:
        return (((p.left == -1) & (p.right == -1)) & (p.name != "root"))

def compute_blengths_from_ages(p):
    if p.ancestor != -1:
        p.blength = p.ancestor.age - p.age
    if p.left != -1:
        compute_blengths_from_ages(p.left)
    if p.right != -1:
        compute_blengths_from_ages(p.right)
    
def set_age1(p,age):
    p.age = age
    highest_age = p.age
    if not istip(p,True):
        h = set_age1(p.left,p.age+p.left.blength)
        if h>highest_age:
            highest_age = h
        h = set_age1(p.right,p.age+p.right.blength)
        if h>highest_age:
            highest_age = h
    return highest_age

def reset_age1(p,age):
    p.age = abs(p.age - age)
    #print(p.age)
    if not istip(p,True):
        reset_age1(p.left,age)
        reset_age1(p.right,age)

class Node:
    """
    Node class: this is a container for content that is saved
    at nodes in a tree
    """
    def __init__(self,pop=-1):
        """
        basic node init
        """
        self.pop = pop
        self.name = ""
        self.left = -1
        self.right = -1
        self.age = 0
        self.ancestor = -1
        self.blength = -1
        self.sequence = []
        self.clean = False
        self.right_seq = []
        self.left_seq = []
        
    def tip(self,name):
        """
        sets the name of a tip
        """
        self.name = name
        self.left = -1
        self.right = -1
        self.ancestor = -1
        self.blength = -1
    
    
    def interior(self,left,right):
        """
        connects an interior node with the two descendents
        """
        self.name = ""
        self.left = left
        self.right = right
        self.ancestor = -1
        self.blength = -1
      
      
    def optimize_branch(self,thetree):
        iterations = 10
        it = 1
        ite = 0
        eps = 0.00001
        b = self.blength
        borig = b
        done = False
        firsttime = True
        while it < iterations and ite < 20 and not done:
            slope, curve, like = centered(b, self, thetree)
            better = False
            if firsttime:
                bold = b
                oldlike = like
                firsttime = False
                better = True
                it += 1
            else:
                if like > oldlike:
                    bold = b
                    oldlike = like
                    better = True
                it += 1
            if better:
                if np.abs(slope)<eps:
                    break
                b = b + slope/np.abs(curve)
                if b < eps:
                    b = eps
            else:
                if np.abs(b - bold) < eps:
                    ite = 20
                b = (b + 19.0 * bold) / 20.
            ite  += 1
            done = np.abs(b-bold) < 0.1*eps
        self.blength = bold
        thetree.lnL = oldlike
        
    def optimize_branch_notwell(self,thetree):
        f = -thetree.likelihood()
        store = self.blength
        if store < 0.000001:
            h = 0.000001
        else:
            h = 0.25 * store
        self.blength = store - h
        if self.blength < 0.0:
            self.blength = 0.0
        f0 = -thetree.likelihood()
        self.blength = store + h
        f2 = -thetree.likelihood()
        if DEBUG:
            print(f'f2={f2} f0={f0} h={h} [f={f}]')
        if f2 == np.inf and f0 == np.inf:
            return
        if np.abs(f2-f0) > 0.000001 and h > 0.000001:
            fdash = (f2 - f0)/h
            self.blength = store - f / fdash
            if self.blength < 0.0:
                self.blength = 0.0
        else:
            fdash = 0.0
            self.blength = store
        x = -thetree.likelihood()
        mult = 0.5
        counter = 0
        while x == np.inf or x > f: # or x < f0 or x < f2:
            self.blength = store - mult * f / fdash
            mult /= 2.0
            if self.blength < 0.0:
                self.blength = 0.0
            x = -thetree.likelihood()
            if np.abs(fdash) < 0.00001 or counter > 20:
                break
            counter += 1


    def myprint(self,file=sys.stdout):
        """
        Prints the content of a node: name if any and branchlengths if any
        """
        if(self.name!="" and self.name!='root'):
            #try:
            #    endname = self.name.index('[')
            #except:
            #    endname = len(self.name)
            #print (self.name[:endname],end='',file=file)
            print (self.name,end='',file=file)
        #print (f"[&age={self.age:.10f}]",end='',file=file)
        if(self.blength != -1):
            print (f":{self.blength:.20f}",end='',file=file)
        elif self.age >= 0.0 and self.ancestor != -1:
            #print(self.ancestor)
            self.blength = self.ancestor.age - self.age    
            #print (":%s" % str(self.blength),end='',file=file)
            print (f":{self.blength:.20f}",end='',file=file)


    def debugprint(self):
        """
        Prints the content of a node: name if any and branchlengths if any
        """
        print (self.name,end=' ')
        print ("(",self.left,end=' ')
        print (self.right,")",end=' ')
        print ("[",self.ancestor,"]",end=' ')
        print (":%s" % str(self.blength),end=' ')
        print ("age=",self.age,end=' ')
        print (self.sequence)

class Tree(Node):
    """
    Node class: this is a container for content that is saved
    at nodes in a tree
    """
    i = 0
    
    
    def __init__(self,root=Node()):
        self.root = root
        self.root.name = "root"
        self.root.blength = 0.0
        ##### self.Q, self.basefreqs = like.JukesCantor()
        self.tree_len=0
    
    
    def myprint(self,p, file=sys.stdout):
        """
        prints a tree in Newick format
        """
        if(p.left != -1):
            print ("(",end='',file=file)
            self.myprint(p.left,file=file)
            print (",",end='',file=file)
        if(p.right != -1):
            self.myprint(p.right,file=file)
            print (")",end='',file=file)
        p.myprint(file=file)
        print ("",end='',file=file)
                
        
    def myprint_multi(self,p, file=sys.stdout):
        """
        prints a tree in Newick format
        """
        if(p.left != -1):
            print ("(",end='',file=file)
            self.myprint_multi(p.left,file=file)
            print (",",end='',file=file)
        if(len(p.right_seq) != 0):
            if(len(p.right_seq) ==1):
                self.myprint_multi(p.right_seq[0],file=file)
                print (")",end='',file=file)
            else:
                for a in p.right_seq[:-1]:
                    self.myprint_multi(a,file=file)
                    print (",",end='',file=file)
                self.myprint_multi(p.right_seq[-1],file=file)    # to remove "," from the last
                print (")",end='',file=file)
        p.myprint(file=file)
        print ("",end='',file=file)

    def treeprint(self,file=sys.stdout):
        if isinstance(file,str):
            myfile = open(file,'w')
            is_open=True
        else:
            myfile = file
            is_open=False
        self.myprint(self.root,myfile)
        print(";",file=myfile)
        if is_open:
            myfile.close()


    #this prints the newick string into a variable
    def treeprintvar(self):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        self.myprint(self.root,file=sys.stdout)
        print(";",file=sys.stdout)
        mynewick = new_stdout.getvalue()
        sys.stdout = old_stdout
        return mynewick

    
    def remove_internal_labels(self,p):
        """
        prints a tree in Newick format
        """
        if p.left != -1 and p.right != -1:
            p.name = ""
        if(p.left != -1):
            self.remove_internal_labels(p.left)
        if(p.right != -1):
            self.remove_internal_labels(p.right)
            
            
    def remove_internal_labels_multi(self,p):
        """
        prints a tree in Newick format
        """
        if p.left != -1 and len(p.right_seq) != 0:
            p.name = ""
        if(p.left != -1):
            self.remove_internal_labels_multi(p.left)
        if(len(p.right_seq) != 0):
            for a in p.right_seq:
                self.remove_internal_labels_multi(a)
            
            
    def myread(self,newick, p):
        """
        reads a tree in newick format
        """
        if(newick[self.i]=="("):
            q = Node()
            p.left = q
            p.left_seq.append(q)
            q.ancestor = p
            self.i += 1
            self.myread(newick,q)

            if(newick[self.i]!=','):      #iteration from here mostly
                print ("error reading, failed to find ',' in %s" % newick)
            self.i += 1
            q = Node()
            p.right = q
            p.right_seq.append(q)
            q.ancestor = p
            self.myread(newick,q)
            
            count=1
            while (newick[self.i]!=')'):
                self.i += 1
                q = Node()
                p.right = q
                if newick[self.i]!=')':
                    p.right_seq.append(q)
                q.ancestor = p
                self.myread(newick,q)
                count +=1
            self.i += 1
        j, p.name = getName(newick[self.i:])
        self.i = self.i + j
        if(newick[self.i] == ":"):
            self.i = self.i+1
            j, xx = getName(newick[self.i:])
            p.blength = float(xx.strip("; \n\t"))
            self.tree_len += p.blength
            self.i = self.i + j

        if newick[self.i] == ';':
            return

    def printTiplabels(self,p):
        if not(istip(p)):
            self.printTiplabels(p.left)
            self.printTiplabels(p.right)
        else:
            print (p.name)
            

    def insertSequence(self, p, label, sequences):
        if not(istip(p)):
            self.insertSequence(p.left,label,sequences)
            self.insertSequence(p.right,label,sequences)
        else:
            the_name = p.name.strip()
            
            if DEBUG:
                print("@@@ tipname", the_name, "Labels:",label)
            try:
                pos = label.index(the_name.strip())
            except:
                try:
                    newlabel = [replace_right(li, "_", "", 1) for li in label]
                    pos = newlabel.index(the_name.strip())
                except:
                    print("Problem with find the label in the list of tips")
                    print(the_name)
                    print(newlabel,label)
                    sys.exit()
            p.sequence = like.tipCondLikelihood(sequences[pos])

    def set_age(self):
        #print(self.root)
        highest_age = set_age1(self.root,0.0)
        reset_age1(self.root,highest_age)

    def condLikelihood(self,p):
        if not(istip(p)):
            self.condLikelihood(p.left)
            self.condLikelihood(p.right)
            p.sequence = like.nodeLikelihood(p.left.sequence,
                                             p.right.sequence,
                                             p.left.blength,
                                             p.right.blength,
                                             self.Q)

    def likelihood(self):
        self.condLikelihood(self.root)
        self.lnL = like.logLikelihood(self.root.sequence,self.basefreqs)
        return self.lnL
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optimize ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def delegate_extract(self,p,delegate):
        if not(istip(p)):
            self.delegate_extract(p.left,delegate)
            self.delegate_extract(p.right,delegate)
            type=''
            if istip(p.left):
                p.left.clean=True
                type+='t'
            else:
                p.left.clean=False
                type+='i'
            if istip(p.right):
                p.right.clean=True
                type+='t'
            else:
                p.right.clean=False
                type+='i'
            p.clean = False
            if p.name=='root':
                type+='r'
            else:
                type+='i'
            delegate.append([p.left,p.right,p, type])

    # def delegate_calclike(self,delegates):
    #     blen=[]
    #     seq =[]
    #     done = False
    #     while not done:
    #         for d in delegates:
    #             if d[0].clean and d[1].clean:
    #                 d[2].sequence = like.nodeLikelihood(d[0].sequence,d[1].sequence,d[0].blength, d[1].blength, self.Q)
    #                 d[2].clean = True
    #                 if d[3][2]=='r':
    #                     done = True
    #                     break
    #             else:
    #                 continue
    #     self.lnL = like.logLikelihood(self.root.sequence,self.basefreqs)
    #     return self.lnL
        
    # def optimizeNR(self):
    #     self.optimize_branch(self.root)
    #     return self.lnL
        
    # def optimize_branch(self,p):
    #     if not(istip(p)):
    #         self.optimize_branch(p.left)
    #         p.optimize_branch(self)   
    #         self.optimize_branch(p.right)
    #         p.optimize_branch(self)
    #     else:
    #         p.optimize_branch(self)   
            
    # def scioptimize(self):
    #     # minimize using Nelder-Mead
    #     delegates=[]
    #     self.delegate_extract(self.root,delegates)
    #     x0,s0,clean0,type0 = extract_delegate_branchlengths(delegates)
    #     bounds = [(0,100)]*len(x0)
    #     result = minimize(neldermeadlike, x0, args=(delegates,self),method='Nelder-Mead',bounds=bounds)
    #     x = result['x']
    #     iterations = result['nit']
    #     print(x,iterations)
    #     instruct_delegate_branchlengths(x,delegates)
    #     self.delegate_calclike(delegates)
    #     if DEBUG:
    #         print("scioptimize: optimize tree likelihood:", self.lnL,iterations)
    #     return(self)

    # def optimize(self):
    #     # minimize using Nelder-Mead
    #     x,fval,iterations, tree1 = opt.minimize_neldermead(self, maxiter=200, maxiter_multiplier=200, initial_simplex=None, xatol=1e-4, fatol=1e-4, adaptive=False, bounds=[0,10])
    #     delegates =[]
    #     tree1.delegate_extract(tree1.root,delegates)
    #     instruct_delegate_branchlengths(x,delegates)
    #     tree1.lnL = -fval
    #     if DEBUG:
    #         print("optimize tree likelihood:", -fval,iterations)
    #     return(tree1)

    # def paupoptimize(self, infile, filetype='RelPHYLIP'):
    #     #create newick
    #     treefile = 'temp_paup_tree'
    #     paupfile = 'temp_paup_tree_optimize.nex'
    #     self.treeprint(file='temp_paup_tree')
    #     write_nexus(paupfile,infile,treefile, filetype)
    #     run_paup(paupfile)
    #     localnewick = phy.readTreeString(treefile+'out')
    #     return(localnewick[-1].strip())
    
    # def setParameters(self,Q,basefreqs):
    #     self.Q = Q
    #     self.basefreqs = basefreqs
    
    # def printLnL(self):
    #     print (self.lnL)
    
    # def getLnL(self):
    #     return self.lnL
    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  get Tips&edges ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def getTips(self,p, tips, tipslength):
        if not(istip(p)):
            for a in p.left_seq:
                tips, tipslength = self.getTips(a, tips, tipslength)
            for a in p.right_seq:
                tips, tipslength = self.getTips(a, tips, tipslength)
        else:
            tips.append(p)
            tipslength.append(p.blength)
        return(tips, tipslength)
    
    
    def get_edges(self,p,edges,edgelengths):
        tips=[]
        tipslength=[]
        if not(istip(p)):
            tips, tipslength = self.getTips(p, tips, tipslength)
            names=[tips[i].name for i in range(len(tips))]
            edges.append(names)
            edgelengths.append(p.blength)
            for a in p.left_seq:
                self.get_edges(a,edges,edgelengths)
            for a in p.right_seq:
                self.get_edges(a,edges,edgelengths)
        return(edges[1:], edgelengths[1:])



    


#==========================   Outside class functions  =======================


#~~~~~~~~~~~~~~~~   Kendal  ~~~~~~~~~~~~~~~~~~
def TreesDistance(m, M, c):      # 0<=c <=1     (For scaling)
    s0=np.sum(np.array(m[0]))
    S0=np.sum(np.array(M[0]))
    s1=np.sum(np.array(m[1]))
    S1=np.sum(np.array(M[1]))
    v1=(1-c)*np.array(m[0])/s0+c*np.array(M[0])/S0
    v2=(1-c)*np.array(m[1])/s1+c*np.array(M[1])/S1
    v11=(1-c)*np.array(m[0])+c*np.array(M[0])
    v22=(1-c)*np.array(m[1])+c*np.array(M[1])
    distance=np.linalg.norm(np.asarray(v1)-np.asarray(v2), ord=2)   #Scaled
#    distance=np.linalg.norm(np.asarray(v11)-np.asarray(v22), ord=2)  #Unscaled
    return distance


def skip_comment(tree):      # to remove the brackets []
    i=0
    tree_new=''
    while i <len(tree):
        if tree[i]=="[":
            while tree[i] != "]":
                i += 1
                tree_new = tree_new +''
        else:
            tree_new = tree_new +tree[i]
        i +=1
    tree_new = tree_new[tree_new.find('=')+1:]
    return tree_new


def generate_treelist(filename, dt):
    data=[]
    Likelihood=[]
    File = open(filename, "r")
    file = File.readlines()
    for line in file:
        temp=line.rstrip().split('\t')
        data.append(temp[-1])
        Likelihood.append(temp[2])
    data=data[1:]
    Likelihood=Likelihood[1:]
    num=int(len(data)/2)
    data = data[num:]
    print("length  of original file = ", len(data))      #length  of data= 78626

    idx=np.arange(0,len(data),dt)
    treelist = [data[i].strip() for i in idx]
    for s, tree in enumerate(treelist):
        tree_new = skip_comment(tree)
        treelist[s] = tree_new
    treelist=[s.replace(' ', '') for s in treelist]
    np.savetxt ('treelist', treelist,  fmt='%s')
    print("length of selected treelist file =", len(treelist))    #length of treelist = 983
    return treelist


#~~~~~~~~~~~~~~~~   random trees  ~~~~~~~~~~~~~~~~~~
def create_random_tree(labels,blens,outgroup=None):
    nodes = labels[:]
    if outgroup != None:
        if outgroup not in nodes:
            print("Warning: outgroup was not correctly specified")
            sys.exit(-1)
    biter = iter(blens)
    rng = default_rng()
    if outgroup != None:
        i = nodes.index(outgroup)
        nodes.pop(i)
    counter = 0
    while len(nodes)>1:
        a,b = rng.permutation(range(len(nodes)))[:2] #rng.integers(low=0, high=len(nodes), size=2)
        la,lb = next(biter),next(biter)
        c = f'({nodes[a]}:{la:.10f},{nodes[b]}:{lb:.10f})'
        counter += 2
        nodes[a] = c
        nodes.pop(b)
    if outgroup != None:
        la,lb = next(biter),next(biter)
        c = f'({nodes[0]}:{la:.10f},{outgroup}:{lb:.10f})'
        nodes[0] = c
        counter += 2
    return nodes[0]+":0.0;"
    
def generate_random_tree(labels,totaltreelength,outgroup=None):
    rng = default_rng()
    a = [1]*(-2 + len(labels)*2)
    blen = totaltreelength * rng.dirichlet(a)
    rt = create_random_tree(labels, blen.tolist(), outgroup)
    return rt

def write_nexus(nexfile,phylipfile,treefile,filetype):
    f = open(nexfile,'w')
    print('begin paup;',file=f)
    print(f'  set crit=likelihood;',file=f)
    print(f'  tonex from={phylipfile} format={filetype} to=/tmp/temp.nex  replace=yes;',file=f)
    print(f'   execute /tmp/temp.nex;',file=f)
    print(f'   gettree file={treefile} warntree=no;',file=f)
    print('    lset nst=1 basefreq=equal;',file=f)
    print('    lscore / replace=yes;',file=f)
    print(f'    savetree / br=yes trees=all format=newick maxdecimals=10 replace=yes file={treefile}out;',file=f)
    print('    quit;',file=f)
    print('end;',file=f)
    f.close()


#~~~~~~~~~~~~~~~~   paup optimize  ~~~~~~~~~~~~~~~~~~
def run_paup(nexfile):
    os.system(f'(paup -n {nexfile}) 2>1 > paup_temp.log')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def centered(b, thenode, thetree):
    eps = 0.000001
    h = 0.01
    if b>eps:
        h = b/2
    if b-h < eps:
        bl=eps
    else:
        bl = b-h
    bu = b + h
    thenode.blength = b
    f = thetree.likelihood()
    thenode.blength = bl
    fl = thetree.likelihood()
    thenode.blength = bu
    fu = thetree.likelihood()
    #print("centered",fu,f,fl)
    slope = (fu - fl)/h
    curve = (fu - 2*f + fl) / (h*h)
    return slope, curve, f
 
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    print("standalone test need to be added!")
