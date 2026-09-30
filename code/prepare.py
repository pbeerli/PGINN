import os
import utils
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
import torch
import numpy as np
import scipy
from torch.utils.data import DataLoader, Dataset
import itertools

def pad(arr,fixed):
    pad_amount = fixed - len(arr)
    #((0, pad_rows), (0, pad_cols)),
    padded_arr = np.pad(arr, ((0, pad_amount),(0,0)), mode='constant', constant_values=0.0)
    return padded_arr

import math
def compute_C0_C1(pairs, n1, n2):
    """
    pairs: list of [age, pop_i, pop_j, i, j]
    returns: (C0, C1)
    """

    # figure out how many tips we actually have
    maxidx = max(max(i, j) for _, _, _, i, j in pairs)
    n_tips = maxidx + 1
    #print(n_tips)
    # infer population of each tip
    #print("up to here")
    tip_pop = [None] * int(n_tips)
    #print("up to here 2")
    for age, pop_i, pop_j, ii, jj in pairs:
        i = int(ii)
        j = int(jj)
        if tip_pop[i] is None:
            tip_pop[i] = pop_i
        elif tip_pop[i] != pop_i:
            #print(tip_pop[i],pop_i)
            raise ValueError(f"Tip {i} has inconsistent pop: {tip_pop[i]} vs {pop_i}")
        if tip_pop[j] is None:
            tip_pop[j] = pop_j
        elif tip_pop[j] != pop_j:
            #print(tip_pop[j],pop_j)
            raise ValueError(f"Tip {j} has inconsistent pop: {tip_pop[j]} vs {pop_j}")
    INF = float("inf")
    min_cross = [INF] * n_tips

    # update minima for cross-pop pairs
    for age, pop_i, pop_j, i, j in pairs:
        if pop_i != pop_j:
            if age < min_cross[i]:
                min_cross[i] = age
            if age < min_cross[j]:
                min_cross[j] = age

    cross0 = [min_cross[k] for k in range(n_tips)
              if tip_pop[k] == 0 and min_cross[k] < INF]
    cross1 = [min_cross[k] for k in range(n_tips)
              if tip_pop[k] == 1 and min_cross[k] < INF]

    def safemean(xs): 
        return sum(xs) / len(xs) if xs else math.nan

    C0 = safemean(cross0)
    C1 = safemean(cross1)
    return C0, C1



def calcC(ll, n1, n2):
    ll = [[float(li[0]),li[1][0],li[1][1]] for li in ll]
    z = 0
    cmin = [0.,0.]
    il1 =[]
    il2=[]
    for i in range(n1+n2):
        for j in range(i+1,n1+n2+1):
            #il1.append(i)
            #il2.append(j)
            ll[z].extend([i,j])
            z += 1
    #print(len(ll[0,:]),ll.size,len(il1))
    #ll = np.stack((ll[0,:],ll[1,:],ll[2,:],il1,il2),axis=1)
    #utils.prettyprint(ll)
    C = compute_C0_C1(ll, n1, n2)
    return C
    
class AveragedThetaDataset(Dataset):
    def __init__(self, folder, poi):
        self.samples = [[],[],[]]
        eps = 1e-8
        print(f"Loading averaged features from: {folder}")
        for filename in sorted(os.listdir(folder)):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(folder, filename)
            #print(f"Reading file: {path}")
            with open(path) as f:
                try:
                    data = json.load(f)
                except Exception as e:
                    print(f"Skipping {path}: {e}")
                    continue

                # Group by theta
                if len(poi)<=2:
                    numpop = 1
                else:
                    numpop = 2
                theta_groups = {}
                for entry in data:
                    theta_tuple = utils.params_to_theta(entry,poi)#PARAM_OF_INTEREST)
                    m = np.array(entry["m"], dtype=np.float32)
                    M = np.array(entry["M"], dtype=np.float32)
                    age = np.zeros_like(M)
                    age_prep = np.array(entry["age"], dtype=np.float32)
                    raw_ages = np.array(entry["ages"], dtype=np.float64) if numpop == 1 else None
                    inter =  entry["inter"]
                    inter_orig = inter
                    if numpop == 2:
                        i01=inter.index([0,1])
                        i11=inter.index([0,0], i01)-i01
                        n1 = i01
                        n2 = i11
                    elif numpop == 1:
                        i01 = len(age)-len(inter)-1
                        i11 = 0
                        n1 = i01
                        n2 = 0
                    else:
                        print("Confusion with number of populations in prepare.py near line 135")
                        sys.exit()
                    z = age_prep.size
                    #print(len(age),len(inter),i01,n1)
                    age[:z] = age_prep[:z]
                    num1 = n1+1
                    # ami0's pad target must cover up to n-1 unique coalescence
                    # ages (one per internal node) plus one slack row; 20 was
                    # only ever "big enough" because every prior dataset had
                    # n<=20 tips (num1==n). max(20,...) keeps n<=20 datasets
                    # (and their already-trained models/scalers) bit-for-bit
                    # unchanged while covering larger n.
                    age_pad_target = max(20, num1)
                    #print(num1)
                    num2 = n2
                    newage = age[:len(age)-num1-num2]
                    xm = m[len(newage):]
                    newage = np.concatenate((newage,xm))
                    p1inter = np.full((num1, 2), 0, dtype=int)
                    newinter = np.array(inter, dtype=int)
                    #print(inter)
                    #print(newinter.shape) #rows, cols = arr.shape
                    if numpop == 2:
                        p2inter = np.full((num2, 2), 1, dtype=int)
                        newinter = np.concatenate((newinter,p1inter,p2inter))
                        big1 = np.stack((newage[:z], M[:z], m[:z], newinter.T[0][:z],newinter.T[1][:z]), axis=1)
                        big2 = np.stack((newage[z:], M[z:], m[z:], newinter.T[0][z:],newinter.T[1][z:]), axis=1)
                        sorted_indices1 = np.lexsort((big1[:,0], big1[:, 4], big1[:, 3]))
                        sorted_indices2 = np.lexsort((big2[:,0], big2[:, 4], big2[:, 3]))
                        sorted_big1 = big1[sorted_indices1]
                        sorted_big2 = big2[sorted_indices2]
                    else: # numpop == 1:
                        newinter = np.concatenate((newinter,p1inter))
                        #print(len(inter), newinter.T[0].shape, z)
                        big1 = np.stack((newage[:z], M[:z], m[:z], newinter.T[0][:z]),axis=1)
                        big2 = np.stack((newage[z:], M[z:], m[z:], newinter.T[0][z:]), axis=1)
                        sorted_indices1 = np.lexsort((big1[:,0],))
                        sorted_indices2 = np.lexsort((big2[:,0],))
                        sorted_big1 = big1[sorted_indices1]
                        sorted_big2 = big2[sorted_indices2]
                    sorted_big = np.concatenate((sorted_big1,sorted_big2))
                    if numpop == 2:
                        age, M, m, inter1, inter2 = sorted_big.T
                        inter = np.stack((inter1, inter2), axis=1)
                    else:
                        age, M, m, inter1  = sorted_big.T
                        inter = inter1
                    #print(age,m,M)
                    #print(age_prep,inter_orig)
                    amix = list(zip(age_prep, inter_orig))
                    #print(amix)
                    C0, C1 = calcC(amix,n1,n2)
                    #print("after C0, and C1")
                    C0 += eps
                    C1 += eps
                    C0 = np.log(C0)
                    C1 = np.log(C1)
                    ratiodir = C0 - C1
                    amix = np.array(sorted([(float(x[0]), *x[1]) for x in amix]))
                    xage = amix[:, 0]
                    xp1 = amix[:, 1]
                    xp2 = amix[:, 2]
                    xpairs = np.column_stack((xp1, xp2))
                    xunique_pairs = np.unique(xpairs, axis=0)
                    xmeans = {}
                    xstd = {}
                    for xpair in xunique_pairs:
                        mask = (xp1 == xpair[0]) & (xp2 == xpair[1])
                        xmeans[(xpair[0], xpair[1])] = np.mean(xage[mask])
                        xstd[(xpair[0], xpair[1])] = np.std(xage[mask])
                    amixset = list(set(map(tuple,amix.tolist())))
                    ami1 = []
                    ami0 = []
                    ami01= []
                    #print("=======")
                    for ai in amixset:
                        if ai[1]==0 and ai[2]==0:
                            ami0.append(ai)
                        if ai[1]==1 and ai[2]==1:
                            ami1.append(ai)
                        if ai[1]==0 and ai[2]==1:
                            ami01.append(ai)
                    #print("+++++")                
                    ami0 = pad(sorted(ami0),age_pad_target)
                    if numpop == 2:
                        ami1 = pad(sorted(ami1),20)
                        ami01 = pad(sorted(ami01),20)
                    #print(f'{ami0=}\n')
                    #sys.exit()
                    xmeans = [list(map(float, [*x[0], x[1]])) for x in list(xmeans.items())]
                    xstd = [list(map(float, [*x[0], x[1]])) for x in list(xstd.items())]

                    if numpop == 2:
                        if theta_tuple not in theta_groups:
                            theta_groups[theta_tuple] = {
                                "m_list": [], "M_list": [],
                                "age_list":[],"agemean_list":[],
                                "agestd_list":[],"a00_list":[],
                                "a11_list":[],"a01_list":[],
                                "C0_list":[], "C1_list":[], "ratio_list":[]}
                        theta_groups[theta_tuple]["m_list"].append(m)
                        theta_groups[theta_tuple]["M_list"].append(M)
                        theta_groups[theta_tuple]["age_list"].append(age)
                        theta_groups[theta_tuple]["agemean_list"].append(xmeans)
                        theta_groups[theta_tuple]["agestd_list"].append(xstd)
                        theta_groups[theta_tuple]["a00_list"].append(ami0)
                        theta_groups[theta_tuple]["a11_list"].append(ami1)
                        theta_groups[theta_tuple]["a01_list"].append(ami01)
                        theta_groups[theta_tuple]["C0_list"].append(C0)
                        theta_groups[theta_tuple]["C1_list"].append(C1)
                        theta_groups[theta_tuple]["ratio_list"].append(ratiodir)
                    else:
                        if theta_tuple not in theta_groups:
                            theta_groups[theta_tuple] = {
                                "m_list": [], "M_list": [],
                                "age_list":[],"agemean_list":[],
                                "agestd_list":[],"a00_list":[],
                                "rawages_list":[]}
                        theta_groups[theta_tuple]["m_list"].append(m)
                        theta_groups[theta_tuple]["M_list"].append(M)
                        theta_groups[theta_tuple]["age_list"].append(age)
                        theta_groups[theta_tuple]["agemean_list"].append(xmeans)
                        theta_groups[theta_tuple]["agestd_list"].append(xstd)
                        theta_groups[theta_tuple]["a00_list"].append(ami0)
                        theta_groups[theta_tuple]["rawages_list"].append(raw_ages)
                        
                for theta, group in theta_groups.items():
                    #print(theta,group)
                    #sys.exit()
                    m_arr = np.stack(group["m_list"])
                    M_arr = np.stack(group["M_list"])
                    m_mean = np.mean(m_arr, axis=0)
                    M_mean = np.mean(M_arr, axis=0)
                    m_std = np.std(m_arr, axis=0)
                    M_std = np.std(M_arr,axis=0)
                    
                    age_arr = np.stack(group["age_list"])
                    age_mean = age_arr.mean(axis=0)
                    age_std = age_arr.std(axis=0)
                    agemean_arr = np.stack(group["agemean_list"])
                    agemean_mean = agemean_arr.mean(axis=0)
                    agemean_std = agemean_arr.std(axis=0)
                    agestd_arr = np.stack(group["agestd_list"])
                    agestd_mean = agestd_arr.mean(axis=0)
                    agestd_std = agestd_arr.std(axis=0)
                    age00_arr = np.stack(group["a00_list"])
                    age00_mean = age00_arr.mean(axis=0)
                    age00_std = age00_arr.std(axis=0)
                    if numpop == 1:
                        rawages_arr = np.stack(group["rawages_list"])
                        rawages_mean = rawages_arr.mean(axis=0)
                    else:
                        rawages_mean = np.zeros(0)
                    if numpop == 2:
                        age01_arr = np.stack(group["a01_list"])
                        temp = np.array(group["a11_list"])
                        age11_arr = np.stack(temp)
                        age11_mean = age11_arr.mean(axis=0)
                        age11_std = age11_arr.std(axis=0)
                        age01_mean = age01_arr.mean(axis=0)
                        age01_std = age01_arr.std(axis=0)
                        C0_arr = np.stack(group["C0_list"])
                        C0_mean = C0_arr
                        C1_arr = np.stack(group["C1_list"])
                        C1_mean = C1_arr
                        ratio_arr = np.stack(group["ratio_list"])
                        ratiodir_mean = ratio_arr                    
                        #print(m_mean.size,M_mean.size,age_mean.size,agemean_mean.size, agestd_mean.size)
                        #print("C0mean",C0_mean)
                        r0 = (age01_mean / ((0.000001 + age00_mean + age11_mean)/2.))[:,0]
                        #print(r0)
                    #sys.exit()
                    if numpop==2:
                        features = np.concatenate((
                                         m_mean,
                                         M_mean,                                         
                                         age_mean,                                         
                                         #agemean_mean.flatten(),
                                         #agestd_std.flatten(),
                                         age00_mean.flatten(),
                                         age11_mean.flatten(),
                                         age01_mean.flatten()
                                         #r0,
                                         #C0_mean,
                                         #C1_mean#,
                                         #ratiodir_mean
                                         ))
                    else:
                        features = np.concatenate((
                                         m_mean,
                                         M_mean,
                                         age_mean,
                                         #agemean_mean.flatten(),
                                         #agestd_std.flatten(),
                                         age00_mean.flatten()
                                         #age11_mean.flatten(),
                                         #age01_mean.flatten(),
                                         #r0,
                                         #C0_mean,
                                         #C1_mean#,
                                         #ratiodir_mean
                                         ))
                    #features = np.concatenate(features)
                    if numpop == 1:
                        # theta_tuple = (theta, s): log-transform theta only.
                        # s can be 0 (neutral case), so it stays on its
                        # natural scale rather than log(0) = -inf.
                        thetas = np.array(theta, dtype=np.float64)
                        thetas[0] = np.log(thetas[0])
                    else:
                        thetas = np.log(theta)

                    #print(f"{thetas=}")
                    #sys.exit()
                    ####self.samples.append((torch.tensor(features, dtype=torch.float32),
                        ###                 torch.tensor(thetas, dtype=torch.float32)))
                    self.samples[0].append(thetas)
                    self.samples[1].append(features)
                    self.samples[2].append(rawages_mean)


        print(f"Saved {len(self.samples)} averaged samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __getitems__(self):
        return self.samples
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Training folder with JSON files")
    parser.add_argument("--interest", type=str, default="0")
    parser.add_argument("--prefix", type=str, default=None)

    args =  parser.parse_args()
    prefix = args.prefix
    traindata = args.train
    PARAM_OF_INTEREST = list(map(int,args.interest.split(',')))
    trainset = AveragedThetaDataset(traindata, PARAM_OF_INTEREST)
    #print(trainset[0])
    sampler = trainset.__getitems__()
    np.save(f'{prefix}-params.npy',sampler[0])
    np.save(f'{prefix}-features.npy',sampler[1])
    np.save(f'{prefix}-ages.npy',sampler[2])

if __name__ == "__main__":
    main()
