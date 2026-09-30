# small utility used in train and predict
import sys
import numpy as np

def prettyprint(arr):
    with np.printoptions(threshold=np.inf, linewidth=np.inf, precision=2, floatmode='fixed'):
        print(arr)

def params_to_theta(entry, poi):
    """
    Convert entry['params'] to a flat theta vector in the *current* spec.
    Adjust indexing here only if the JSON spec changes again.

    Example old: [[theta1, theta2], [0, M21, M12, 0], [1,1]]
    Example new:  read list and create the correct block
    """
    P = entry["params"]
    #print(len(P), P)
    #if len(poi) < len(P):
    #print(poi)
    #print(P)
    theta_tuple = tuple((np.array(P).flatten())[pix] for pix in poi)
    #print(theta_tuple)
    #print(P)
    #print(poi)
    #sys.exit()
    ##if len(P) == 1:
        #single population
    ##   theta_tuple = tuple(map(float,P[0]))
    ##elif len(P)>= 2:
        #print(P[0])
        #print(P[1])
        #print(P)
        #theta_tuple = tuple(map(float,P[0]),map(float,P[1]))
     #   theta_tuple = tuple((tuple(map(float,P[0])),tuple(map(float,P[1]))))
        #print(theta_tuple)
    #else:
    #    print("Problem in util.py",P)
    #    sys.exit()
    #theta_tuple = (float(P[0][0]), float(P[0][1]),
    #               float(P[1][0]), float(P[1][1]))
    #print("utils:",theta_tuple)
    return theta_tuple
