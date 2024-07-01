import numpy as np
import sys

def remove_unactive_links_backward(current_adj, after_adj):
    outdegree = np.sum(after_adj, axis=1)
    current_one = current_adj.copy()
    outdegree[outdegree>0] = 1
    current_adj = current_adj * outdegree

    print("Number of removed unactive links backwards: ", int(np.sum(current_one) - np.sum(current_adj)))

    return current_adj

def remove_unactive_links_forward(current_adj, before_adj):
    indegree = np.sum(before_adj, axis=0)
    current_one = current_adj.copy()
    indegree[indegree>0] = 1
    current_adj = current_adj * indegree.reshape(-1, 1)

    print("Number of removed unactive links forwards: ", int(np.sum(current_one) - np.sum(current_adj)))

    return current_adj
