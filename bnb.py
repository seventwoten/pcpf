import numpy as np
from math import cos, sin
    
def subdivide(x, y, z, sigma):
    '''Returns 8 block centres and new half-side length, sigma'''
    shifts = np.array([[-1, -1, -1],
                       [-1, -1,  1],
                       [-1,  1, -1],
                       [-1,  1,  1],
                       [ 1, -1, -1],
                       [ 1, -1,  1],
                       [ 1,  1, -1],
                       [ 1,  1,  1]])
    
    return np.array([[x,y,z]]) + sigma/2 * shifts, sigma/2    

def aa2mat(b_xyz):
    '''Given 3d point in R-space ball, returns R matrix'''
    alpha = np.linalg.norm(b_xyz)
    u = b_xyz/alpha
    
    c = cos(alpha)
    s = sin(alpha)
    t = 1-c

    U_outer = np.dot(u.T, u)
    U_as = np.array([[0, -u[0][2], u[0][1]],[u[0][2], 0, -u[0][0]],[-u[0][1], u[0][0], 0]])
    
    return c * np.eye(3) + t * U_outer + s * U_as
    
    
def MaxFlow(C, s, t):
    '''Returns cardinality of maxflow (push-relabel code is copied)
       Copied from http://www.cs.sjsu.edu/faculty/pollett/masters/Semesters/Fall16/sarika/Push-RelabelAlgorithm.txt 
       (Will be customised later)
    '''
    
    n = len(C) # C is the capacity matrix
    F = [[0] * n for i in range(n)]

    # the residual capacity from u to v is C[u][v] - F[u][v]
    height = [0] * n # height of node
    excess = [0] * n # flow into node minus flow from node
    seen   = [0] * n # neighbours seen since last relabel
    # node "queue"
    nodelist = [i for i in range(n) if i != s and i != t]

    #push operation
    def push(u, v):
        send = min(excess[u], C[u][v] - F[u][v])
        F[u][v] += send
        F[v][u] -= send
        excess[u] -= send
        excess[v] += send

    #relabel operation
    def relabel(u):
        # find smallest new height making a push possible,
        # if such a push is possible at all
        min_height = float('inf')
        for v in range(n):
            if C[u][v] - F[u][v] > 0:
                min_height = min(min_height, height[v])
                height[u] = min_height + 1
 
    def discharge(u):
        while excess[u] > 0:
            if seen[u] < n: # check next neighbour
                v = seen[u]
                if C[u][v] - F[u][v] > 0 and height[u] > height[v]:
                    push(u, v)
                else:
                    seen[u] += 1
            else: # we have checked all neighbours. must relabel
                relabel(u)
                seen[u] = 0
 
    height[s] = n   # longest path from source to sink is less than n long
    excess[s] = float("inf") # send as much flow as possible to neighbours of source
    for v in range(n):
        push(s, v)
 
    p = 0
    while p < len(nodelist):
        u = nodelist[p]
        old_height = height[u]
        discharge(u)
        if height[u] > old_height:
            nodelist.insert(0, nodelist.pop(p)) # move to front of list
            p = 0 # start from front of list
        else:
            p += 1
    return sum(F[s])