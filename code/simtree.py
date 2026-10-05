#!/usr/bin/env python
# -*- coding: utf-8 -*-
# simtree-ML-date.py
# a variant of simtree-ML that allows to add dated samples
# @Time    : 2023/2/23 11:47    
# @Author  : Peter Beerli
# @Site    : Tallahassee, Florida, USA
#
# added JSON for machine learning training
# added dating of samples
#
# updated  early August 2025: fixed and error that lead to
# short time intervals: the error was a logic problem:
# 1. pick smallest time for all individual events (coalescence, migrations)
# 2. pick event by calculating CDF for events and choose one according to
#    the probability -- then assign that event
# (this lead to an error of a factor 4x shorter time intervals
# now fixed:
# 1. pick smallest time for all individual events and record which event
#    is associated with the smallest event.f
#
# This copy is customized for the PGINN paper (manuscript-2): it adds
# --timepoints (time-varying theta for the selective-sweep model), --seed,
# --selection and the internal-node ages in the JSON output. It is kept
# here instead of tracking the general simulator
# (github.com/pbeerli/fractcoalsim), which does not have these options.
#
# updated October 5 2026: migration rates fixed as in fractcoalsim
# commit 9f18526 (migration_matrix()). mylambda() read M[j], only the
# first row of the -m list: "a,b" simulated M_2->1 = b and M_1->2 = a,
# and "0,a,b,0" simulated M_2->1 = a and M_1->2 = 0. The paper uses one
# population only, where no migration rate is read, so its results are
# unchanged.
# 2. assign event
# Nov 2025 added multiple timepoints for changes of population size -- works only for a single population
#
import os
import io
import json as jason
import matplotlib.pyplot as plt
import math
import numpy as np
import Mm as mm
import sys
#print (sys.path)
#sys.exit()
np.math = math
#from numba import jit
plt.rcParams['text.usetex'] = True

import tree
#import numpy.ma as ma

def pick_two_lineages(population,pop,kpop):
    #x = ma.array(population, mask=ma.masked_not_equal(population, pop))  
    x = [i for i in range(len(population)) if population[i].pop == pop]
    y = [i for i in range(len(population)) ]
    #print("x:", [population[xi].pop for xi in x],len(population),kpop) 
    #print("y:", [population[xi].pop for xi in y],len(population) )  
    i,j = np.random.choice(x, size=2, replace=False)
    return i,j

def pick_one_lineage(population, pop):
    x = [i for i in range(len(population)) if population[i].pop == pop]
    i = np.random.choice(x, size=1, replace=False)
    return i

def setup_population(samples):
    numberpopulations = len(samples)
    #print(samples)
    populations = []
    for pop in range(numberpopulations):
        for i in range(samples[pop]):
            xx = tree.Node(pop)
            xx.name = str(pop) + '_' + str(i)
            xx.age = 0.0
            populations.append(xx)
    #print(populations)
    return populations   

def add_samples(age, populations, k, newsamples):
    numberpopulations = len(newsamples)
        
    for pop in range(numberpopulations):
        k[pop] += newsamples[pop]
        for i in range(newsamples[pop]):
            xx = tree.Node(pop)
            xx.name = str(pop) + '_'+str(age)+'_' + str(i)
            xx.age = age
            populations.append(xx)
    #[print(pop.age) for pop in populations]
    return populations   


def connect_lineages(population, i, j, age ):
    #mypopulation = population[:]
    #print("population:", population)
    #print("ind:", population[i].pop, population[j].pop)
    q  = tree.Node()
    q.pop = population[i].pop 
    q.age = age
    population[i].ancestor = q
    population[j].ancestor = q
    q.left = population[i]
    q.right = population[j]
    population[i] = q 
    population.pop(j)
    return population

def migration_matrix(mlist, npop):
    """Full migration matrix Mf[i][j] = M_{j->i} (forward in time: migrants
    into population i from population j; backward, a lineage in i moves to
    j), from the -m list in migrate's order: row by row, row i = receiving
    population. Accepted: npop*npop values (diagonal ignored), the
    npop*(npop-1) off-diagonal values in the same order, or one value for
    all. For two populations "0,a,b,0" and "a,b" both mean M_2->1 = a,
    M_1->2 = b."""
    Mf = [[0.0] * npop for _ in range(npop)]
    if npop < 2:
        return Mf
    n = len(mlist)
    if n == 1:
        vals = [mlist[0]] * (npop * (npop - 1))
    elif n == npop * npop:
        vals = [mlist[i * npop + j] for i in range(npop) for j in range(npop) if i != j]
    elif n == npop * (npop - 1):
        vals = list(mlist)
    else:
        raise SystemExit(f"-m needs 1, {npop*(npop-1)} or {npop*npop} values for {npop} populations, got {n}")
    z = 0
    for i in range(npop):
        for j in range(npop):
            if i != j:
                Mf[i][j] = vals[z]
                z += 1
    return Mf

#@jit(nopython=True)
def mylambda(theta,M,k,alphas):    
    npop = len(theta)
    npop2 = npop**2
    #if alpha==None:
    #    alphas = np.ones(npop)
    #else:
    #    alphas = np.array(alpha,dtype=np.float64)     
    L = fill_mygammas(alphas)
    #print("gammas:",np.exp(l))
    xlambdas = np.zeros(npop2)
    for i in range(npop):
        if k[i]> 1:
            xlambdas[i] =   k[i] * ( k[i]-1 ) / theta[i]
        else:
            xlambdas[i] = 0.0
    z = npop
    for i in range(npop):
        for j in range(npop): 
            if i != j:
                xlambdas[z] = M[i][j]*k[i]   # M = migration_matrix(): M_{j->i}
                z += 1
    Y = xlambdas*np.exp(L)
    #print("k:",k)
    #print("xlambdas:",xlambdas)
    #print("Y:",Y)
    return Y,xlambdas,L



#@jit(nopython=True)
def lgamma(x):
    return np.log(np.math.gamma(x))

#@jit(nopython=True)
def fill_mygammas(alphas):
    npop = len(alphas)
    npop2 = npop*npop
    mygammas = np.zeros(npop2)
    allgamma=0
    for i in range(npop):
        allgamma += lgamma(alphas[i]+1)
    z = npop
    for i in range(npop):
        # missing a term? mygammas[i] = allgamma - lgamma (alphas[i]+1)
        mygammas[i] = - lgamma (alphas[i]+1) #allgamma - lgamma() - allgamma
        #print(f"mygammas[{i}]={mygammas[i]}")
        for j in range(npop): 
            if i != j:
                #mygammas[z] = (allgamma - lgamma(alphas[i]+1)) - allgamma
                mygammas[z] = - lgamma(alphas[i]+1)
                #print(f"mygammas[{z}]={mygammas[z]}")
                z += 1  
    return mygammas

#@jit(nopython=True)
def fill_Yalphas(alphas):
    npop = len(alphas)
    npop2 = npop**2
    Yalphas = np.zeros(npop2)
    z = npop
    for i in range(npop):
        Yalphas[i] = alphas[i]
        for j in range(npop): 
            if i != j:
                Yalphas[z] = alphas[i]
                z += 1  
    return Yalphas


#Somayeh Mashayekhi                                                           
#double propose_new_mlftime(double lambda, double alpha, double r1, double r2)
#{
#  double pia = PI * alpha;
#  double denoma = 1.0 / alpha;
#  double denomlambda = 1.0 / lambda;
#  return -pow(denomlambda,denoma) * pow((sin(pia)/(tan(pia*(1.0-r1))) - cos(pia\
#)),denoma) * log(r2);
#  //return -pow(denomlambda * (sin(pia)/(tan(pia*(1-r1))) - cos(pia)),denoma) *\
# log(r2);                                                                       
#}

#@jit(nopython=True)
def randommlftime(mylambda, alpha):
    pia = 3.1415926 * alpha
    r1 = np.random.uniform(0,1)
    r2 = np.random.uniform(0,1)
    denoma = 1.0 / alpha
    if mylambda == 0.0:
        return 1e100
    denomlambda = 1.0 / mylambda
    return -denomlambda**denoma * (np.sin(pia)/(np.tan(pia*(1.0-r1)))-np.cos(pia))**denoma * np.log(r2)

#@jit(nopython=True)
def randomtime(Y,alphas,t0):
    smallu = 1e100
    smalli = -1
    for i,yi,ai in zip(range(len(Y)),Y,alphas):
        u = randommlftime(yi,ai)
        #print(f"{u=} kingman:{kingmanrandomtime(Y,t0)} {yi=} {ai=}")
        if u < smallu:
            smallu = u
            smalli = i
    return t0 + smallu, smalli

#@jit(nopython=True)
def kingmanrandomtime(Y,t0):
    #lY = len(Y)
    kY = np.sum(Y)
    #Y = Y + 1e-10
    
    #u =  (np.log(Y) - np.log(np.random.uniform(0,1,lY)))/Y 
    #u =  ( - np.log(np.random.uniform(0,1,lY)))/Y
    u =  ( - np.log(np.random.uniform(0,1)))/kY 
    #u[u<=0] = 1000000
    return t0 + u

#@jit(nopython=True)
def oldrandomtime(Y,t0):
    lY = len(Y)
    Y = Y + 1e-10
    #u =  (np.log(Y) - np.log(np.random.uniform(0,1,lY)))/Y 
    u =  ( - np.log(np.random.uniform(0,1,lY)))/Y 
    u[u<=0] = 1000000
    return t0 + u

#@jit(nopython=True)
def mm2m(i,numpop):
    #print(i)
    if i<numpop:
        return i, i
    else:
        topop = (i - numpop) // (numpop - 1)
        frompop = i - numpop - topop * (numpop - 1)
        if frompop >= topop:
            frompop += 1;
        return int(frompop),int(topop)

def pick_event(lambdas):
    s = np.sum(lambdas)
    x = lambdas/s
    #print(x)
    return np.random.choice(range(len(lambdas)), p=x)

    
import sys 
#@jit(nopython=True)
def sim_one(ne,M,savek,alpha,numpop,sampledates,timepoints):
    global fullk
    populations = setup_population(savek)
    age = 0.0
    k = savek.copy()
    count = 0
    #otmin=0.0
    mytimes=np.zeros(100000)
    while np.sum(k)>1 and count < 100000:
        count += 1
        Y,debuglambdas,debugl = mylambda(ne,M,k,alpha)
        Yalphas = fill_Yalphas(alpha)
        t1, which = randomtime(Y,Yalphas, age)
        #if DEBUG:
        #print("@", count, age, t1,k, Y, debuglambdas,np.exp(debugl),file=sys.stderr)
        if len(sampledates)>0 and t1 > sampledates[0][0]:
                newsamples = sampledates.pop(0)
                t1 = newsamples[0]
                u = 0
                #for pop in populations:
                #    pop.age = t1
                populations = add_samples(t1, populations, k, newsamples[1])
                #print(len(populations), k)
                #print(f"@@{t1} {t1-age} {t1-u} {frompop} :: {k}  {[i.pop for i in populations]}")
                age=t1
                continue
        elif len(timepoints)>0 and t1 > timepoints[0][0]:
            newtimepoints = timepoints.pop(0)
            t1 = newtimepoints[0]
            u=0
            param = newtimepoints[1]
            if len(param)==1:
                ne = param
            #print(param, ne)
            #sys.exit()
            #print("New theta for simulation",ne)
            
            age = t1
            continue
        else:
            u = t1 - age
            age = t1
            #print("@@@@", age)
                    
                #change all in populations to time
        #print("time choices:", t1,end=' | ')
        #mini = np.argmin(t1)
        #tmin = t1[mini]
        #tmin = t1
        #u = tmin - otmin
        #mytimes[count] = u
        #########mytimes[count] = age
        #otmin = tmin
        #otmin = age
        mini=which #pick_event(debuglambdas)
        #print("index:",mini, "time:",tmin, "interval:",u)
        frompop, topop = mm2m(mini,numpop)
        if frompop == topop:
            #print(populations[frompop])
            if k[frompop] < 2:
                #print("k[frompop] < 2")
                break
            l1,l2 = pick_two_lineages(populations,frompop,k[frompop])
            #print("pick:",frompop,"::", l1,l2)
            populations = connect_lineages(populations,l1,l2, age)
            k[frompop] -= 1
        else:
            l1 = pick_one_lineage(populations,topop)[0]
            #print("pick2:",topop,"::", l1)
            populations[l1].pop = frompop              
            k[topop] -= 1
            k[frompop] += 1   
        ###@@print(f"{age} {u} {age-u} {frompop} :: {k}  {[i.pop for i in populations]}")
    if len(sampledates)>0:
        print(f"WARNING: not all sampledates are used: remaining {sampledates=}")
        for s in sampledates:
            fullk -= np.array(s[1])
    return populations  

#@jit(nopython=True)
def run_sim(nrun, ne,M,k,alpha,alpha2, sampledates):
    numpop = len(ne)
    simtimes=np.zeros(nrun)
    simtimes2=np.zeros(nrun)
    for i in np.arange(0,nrun): 
        simtimes[i],count = np.sum(sim_one(ne,M,k,alpha,numpop, sampledates, timepoints))
        simtimes2[i],count2 = np.sum(sim_one(ne,M,k,alpha2,numpop, sampledates, timepoints))
        if count>=LARGE or count2>=LARGE:
            sys.exit(f'Did not coalesce within {LARGE} time steps')

    return simtimes,simtimes2

def collect_internal_ages(p, ages):
    """Appends the ages of all internal (coalescent) nodes below and including p."""
    if not tree.istip(p, True):
        ages.append(p.age)
        collect_internal_ages(p.left, ages)
        collect_internal_ages(p.right, ages)

def report_stats(simtimes, loci, title=''):
    print(title)
    print("TMRCA         = ",np.mean(simtimes))
    print("TMRCA:std     = ",np.std(simtimes))
    print("TMRCA:min,max = ",np.min(simtimes),np.max(simtimes))   
    print("Loci          = ",loci)


import argparse as ap

if __name__ == '__main__':
    DEBUG = False
    LARGE = 100000    
    parser = ap.ArgumentParser(description='Simulate a tree')
    parser.add_argument('-id', '--ID', type=int, default=1, help='ID number start, each locus will add +1')
    parser.add_argument('-l', '--loci', type=int, default=1, help='number of loci')
    parser.add_argument('-s', '--sites', type=int, default=1000, help='number of sites')
    parser.add_argument('-i', '--individuals', type=str, default="10,10", help='Number of samples for each population')    
    parser.add_argument('-t', '--theta', type=str, default="0.01,0.01", help='thetas for each population')    
    parser.add_argument('-m', '--mig', type=str, default="0,100,100,0", help='migration rates M (mutation-scaled) in migrate order, row i = receiving population: for 2 populations "0,M_2->1,M_1->2,0" (diagonal ignored) or "M_2->1,M_1->2"; one value for all')
    parser.add_argument('-a', '--alpha', type=str, default="1.0,1.0", help='alpha for each population')
    parser.add_argument('-f', '--file', type=str, default="NONE", help='treefile to be used with migdata, default is NONE which is a placeholder for sys.stdout')
    parser.add_argument('-p', '--plot', action='store_true', help='Plots density histogram of TMRCA')
    parser.add_argument('-j', '--json', action='store_true', help='uses a json output style')
    parser.add_argument('-d', '--datefile', type=str, default=None, help='allows for dated samples in file: time sample1 sample2 [units: theta]')
    parser.add_argument('-tp', '--timepoints', type=str, default=None, help='allows for n timepoints that change theta read from file timepoints, that is organized per line time theta; time is in theta units" ')
    parser.add_argument('-wd', '--writedatefile', type=str, default=None, help='write datefile for migrate')
    parser.add_argument('--seed', type=int, default=None, help='numpy RNG seed; omit for a nondeterministic run')
    parser.add_argument('-sel', '--selection', type=float, default=0.0, help='true selection coefficient s used to build --timepoints (bookkeeping only, recorded into the JSON params; does not affect the simulation directly)')
    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    myfile = args.file
    sites = args.sites
    loci = args.loci    
    plot = args.plot
    json = args.json
    ID = args.ID
    dated = args.datefile
    timepointfile = args.timepoints
    timepoints = []
    writedatefile = args.writedatefile
    selection = args.selection
    sampledates = []
    if dated!=None:
        with open(dated,'r') as f:
            for line in f:
                if line[0]=='#':
                    continue
                else:
                    tmp = line.strip().split()
                    t = float(tmp[0])
                    inds = list(map(int,tmp[1:]))
                    sampledates.append([t,inds])
        #print(f"{sampledates=}")
    if timepointfile!=None:
        with open(timepointfile,'r') as f:
            for line in f:
                if line[0]=='#':
                    continue
                else:
                    tmp = line.strip().split()
                    t = float(tmp[0])
                    thetas = list(map(float,tmp[1:]))
                    timepoints.append([t,thetas])
        #print(f"{timepoints=}")
    #sys.exit()
    
    #individuals today, check whether there is a dated data with time 0,
    #then use that instead
    k=[int(ki) for ki in args.individuals.split(',')]
    if sampledates!=[] and sampledates[0][0] == 0.0:
        k = sampledates.pop(0)[1]
    #print(k)
    fullk = np.array(k)
    for s in sampledates:    
        fullk += np.array(s[1])
    #print(f"{k=}")
    #print(f"{fullk=}")
    #[10,10]
    ne = [float(ki) for ki in args.theta.split(',')]
    npop = len(ne)
    #np.array([0.01,0.01]) 
    M = migration_matrix([float(ki) for ki in args.mig.split(',')], npop)
    #print(f"{M=}")
    #np.array([0,100.,100.,0])
    alpha = [float(ki) for ki in args.alpha.split(',')]   
    #np.array([0.9,0.9])
    alpha2 = np.array([1.0]*len(ne))
    if myfile == 'NONE':
        thetree = sys.stdout 
    else:
        thetree = open(myfile,'w')
    nl=os.linesep
    simtimes=[] 
    simtimes2=[]
    r = np.random.randint(1000000)
    
    if not json:
        kkstr = " ".join(map(str,fullk))
        thetree.write(f"#SN{nl}#{r}{nl}#{len(ne)}{nl}#{kkstr}{nl}#{loci} {sites} 2.0{nl}")
        #thetree.write(f"#SN{nl}#{r}{nl}#2{nl}#{k[0]} {k[1]}{nl}#{loci} {sites} 2.0{nl}")
        for locus in range(loci):
            thetree.write(f"# rate among sites for locus 0 (1.000000){nl}#={nl}")
    data = []
    for locus in range(loci):
        if not json:
            thetree.write(f"#$ locus {locus}{nl}#$ 0.000001{nl}")
        # sim_one() pops entries off `timepoints` as it consumes them, so a
        # fresh copy is needed per locus -- otherwise the schedule is silently
        # depleted after the first locus and every later locus in this run
        # sees no (or a truncated) time-varying theta.
        root = sim_one(ne,M,k,alpha, npop,sampledates, list(timepoints))
        if plot:
            root2 = sim_one(ne,M,k,alpha2,2,sampledates, list(timepoints))
            simtimes2.append((root2[0].age))
        t = tree.Tree(root[0])
        simtimes.append((root[0].age))
        tree.compute_blengths_from_ages(t.root)
        t.set_age()
        if writedatefile:
            with open(writedatefile,'w') as datefilef:
                t.printTiplabels(t.root, file=datefilef)
                
        if not json:
            t.myprint(t.root,file=thetree)
            thetree.write(f";{nl}")
        else:
            buffer = io.StringIO()
            t.myprint(t.root,file=buffer)
            buffer.write(f";")
            newick = buffer.getvalue()
            mmt = mm.mmtree(t.root)
            pairlist = []
            tips = []
            tipslength = []
            a = mmt.MetricsVectors(newick, tips, tipslength, pairlist)

            internal_ages = []
            collect_internal_ages(t.root, internal_ages)
            internal_ages = sorted(internal_ages, reverse=True)

            xxx = [ne, M, alpha]
            if npop==1:
                xxx = [ ne, alpha, [selection]
                       ]
            jsondata= {
                "ID": ID,
                "params" : xxx,
                "M": a[0],
                "m": a[1],
                "age" : a[2],
                "inter" : a[3],
                "ages" : internal_ages,
                "tree" : newick
            }
            ID += 1
            #data.append(jason.dumps(jsondata, indent=4))
            data.append(jsondata)

    report_stats(simtimes,loci,f'{alpha}')            
    if not json:
        if myfile != 'NONE':
            thetree.close()
    else:
        if myfile != 'NONE':
            jason.dump(data, thetree, indent=0)
            
            #for d in data:
            #    thetree.write(str(d))
            #    thetree.write("\n")
            #thetree.close()
        else:
            for d in data:
                print(d)
                
    #simtimes,simtimes2 = run_sim(1000, ne,M,k,alpha,alpha2)
    if plot:
        report_stats(simtimes2,loci,f'{alpha2}')  
        plt.hist(simtimes, bins=50,color='blue', histtype='step', density=True, alpha=0.6, label=r'$\alpha=$'+f'{alpha}')
        plt.hist(simtimes2, bins=50, density=True, alpha=0.6, color='red',histtype='step', label=r'$\alpha=1.0$')
        plt.legend(loc='upper right')
        plt.xlim(0,0.12)
        plt.ylim(0,80.0)
        plt.xlabel('Time to MRCA')
        plt.ylabel('Density')
        plt.savefig('simtreefig.pdf')
    
    
