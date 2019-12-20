#!/usr/bin/env python
# coding: utf-8

# In[15]:


import networkx as nx
import numpy as np
from scipy.optimize import fsolve
from progressbar import ProgressBar


# In[16]:


def DTPpmf(x,alpha,beta):
    return ((1/(1-(1/(beta + 1))**alpha))*(1/(x**alpha) - 1/((x+1)**alpha)))
    
def discTruncPareto(n,xc,c):
    from progressbar import ProgressBar
    # generate a discretised pareto distribution truncated at n 
    # that accounts for c proportion of blocks from xc proportion of nodes
    
    func = lambda alpha : (1-(1/(xc*n))**alpha) / (1-(1/(n+1))**alpha) - c

    # Use the numerical solver to find the roots

    alpha_initial_guess = 0.5
    alpha_solution = fsolve(func, alpha_initial_guess)
    alpha_solution
    
    # generate pmf
    pmf = np.zeros(n)
    for i in range(n):
        pmf[i] = DTPpmf(i+1,alpha_solution,n)
    return(pmf)


# In[17]:


def createGraph(degree,normal,testing,validation):
    # number of nodes
    n = normal
    # mean degree
    deg = degree


    # number of abberant nodes
    numtestnodes = testing
    numvalnodes = validation

    nodes = range(n)
    valnodes = range(n,n+numvalnodes)
    testnodes = range(n + numvalnodes,n+numvalnodes+numtestnodes)
    total = n+numvalnodes+numtestnodes

    # number of edges
    m = total*deg/2
    
    # normal nodes
    G=nx.gnm_random_graph(total,m)


    
    # parameters for each node's queuing and processing times
    expParameters = np.zeros(total)
    unifParameters = np.zeros(total)
    for i in range(total):
        # exponential delay parameter from Gamma(20,5)
        expParameters[i] = np.random.gamma(20,5)
        # uniform processing parameter from Pareto(1.25,20)
        unifParameters[i] = (np.random.pareto(1.25)+1)*20

    return(G,nx.to_dict_of_lists(G),expParameters,unifParameters)


# In[18]:


def cascade(n,total,pmf,delaypar,processpar,AdjDict):
    # create arrival times matrix
    label = np.arange(total)
    arrival = np.ones(total)*1000000
    measure = np.zeros(total)
    times = np.stack([label,arrival,measure],axis=1)
    # generate source node
    source = np.random.choice(np.arange(n),p = pmf)
    times[source,1] = 0
    # sort arrival times
    times = np.array(sorted(times, key=lambda x: x[1]))

    for i in range(total):
        node = times[i,0]
        basetime = times[i,1]
        peers = AdjDict[times[i,0]]
        # create measurement
        times[i,2] = basetime + np.random.exponential(delaypar[int(node)])
        # calculate arrival times at peers
        times = np.array(sorted(times, key=lambda x: x[0]))
        for j in peers:
            # ignore if arrival time for peer is earlier
            if times[j,1] > basetime:
                delaytime = np.random.exponential(delaypar[int(node)])
                processingtime = np.random.uniform(20,processpar[j])
                newarrivaltime = basetime + delaytime + processingtime
                # update arrival time
                if newarrivaltime < times[j,1]:
                    times[j,1] = newarrivaltime
        # sort times
        times = np.array(sorted(times, key=lambda x: x[1]))
    
    return(times, source)


# In[19]:


# cascades to perform
tests = 1000

# graph details
meandegree = 8
normal = 200
validation = 5
testing = 5

total = normal+testing+validation

n = normal
v = validation
t = testing
tot = n+v+t

# block generation distribution xc proportion of nodes accounts for c proportion of blocks
xc = 0.10
c = 0.50
pmf = np.array(discTruncPareto(normal,xc,c))

# generate Graph
Graph, AdjDict, delaypar, processpar = createGraph(meandegree,normal,testing,validation)

ValAdj = {k:AdjDict[k] for k in range(normal, normal + validation)}
TestAdj = {k:AdjDict[k] for k in range(normal + validation, total)}

edges = 0
for i in AdjDict:
    edges += len(AdjDict[i])
edges = edges/2


# In[20]:


# record cascade data
data = np.zeros([total,tests])
sources = np.zeros(tests)
pbar = ProgressBar()
for i in pbar(range(tests)):
    times, source = cascade(normal,total,pmf,delaypar,processpar,AdjDict)
    sources[i] = source
    times = np.array(sorted(times, key=lambda x: x[0]))
    newdata = times[:,2]
    data[:,i] = newdata

# create correlation matrix
corrmatrix = np.corrcoef(data)


# In[21]:


np.savetxt("delay_data.csv", data, delimiter=",")
np.savetxt("corr_matrix.csv", corrmatrix, delimiter=",")

