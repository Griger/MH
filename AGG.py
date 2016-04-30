import numpy as np
from knnLooGPU import *

def cross(first_parent, second_parent, first_son, second_son):
    not_equals_idx = first_parent ^ second_parent
    equals_idx = np.logical_not(not_equals_idx)

    #copy identical genes into children
    first_son = first_parent & equals_idx
    second_son = first_parent & equals_idx

    #random different genes selection
    not_equals_size = sum(not_equals_idx)
    first_son[not_equals_idx] = np.random.choice([True, False], not_equals_size)
    second_son[not_equals_idx] = np.logical_not(first_son[not_equals_idx])

def mutate(s, gen_idx):
    s[gen_idx] = not s[gen_idx]

def AGG(train_data, train_labels, knnGPU):
    n = len(train_data[0])
    p_size = 30 #population size

    size_chromosome_string = str(n) + 'bool'
    datatype = np.dtype( [('chromosome',size_chromosome_string), ('score',np.float32)] )

    parent = np.zeros(p_size, dtype=datatype)
    parent["chromosome"] = np.random.choice([True,False], (p_size, n)) #random initial population

    for individual in parent:
        individual["score"] = knnGPU.scoreSolution(train_data[:,individual["chromosome"]], train_labels)

    parent.sort(order="score")


    #selection by binary tournament
    selected_parent_idx = np.empty(p_size, dtype=np.int32)

    for idx in selected_parent_idx:
        idx = np.random.randint(np.random.randint(0,p_size), p_size)

    children = np.zeros(p_size, dtype=datatype)

    for first_parent_idx, second_parent_idx, first_son, second_son in idx[::2], idx{1::2], children[::2], children[1::2]:
        cross(parent[first_parent_idx], parent[second_parent_idx], first_son, second_son)

    for son in children:
        son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

    children.sort(order="score")
