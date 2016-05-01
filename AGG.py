import numpy as np
from knnLooGPU import *
from math import *

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

def isSolHere(sol, sols_set):
    equal_elements = sol == sols_set
    return len(s) in equals_elements.sum(axis = 1)

def AGG(train_data, train_labels, knnGPU):
    max_evals = 15000
    n = len(train_data[0])
    p_size = 30 #population size

    cross_p = 0.7
    n_crosses = ceil(p_size/2 * 0.7)

    mutation_p = 0.001
    n_mutations = ceil(p_size * n * mutation_p)

    n_evals = 0

    size_chromosome_string = str(n) + 'bool'
    datatype = np.dtype( [('chromosome',size_chromosome_string), ('score',np.float32)] )

    parent = np.zeros(p_size, dtype=datatype)
    parent["chromosome"] = np.random.choice([True,False], (p_size, n)) #random initial population

    for individual in parent:
        individual["score"] = knnGPU.scoreSolution(train_data[:,individual["chromosome"]], train_labels)

    n_evals += p_size

    parent.sort(order="score")

    while (n_evals < max_evals):
        #selection by binary tournament
        selected_parent_idx = np.empty(p_size, dtype=np.int32)

        for idx in selected_parent_idx:
            idx = np.random.randint(np.random.randint(0,p_size), p_size)

        #cross
        children = np.zeros(p_size, dtype=datatype)

        for first_parent_idx, second_parent_idx, first_son, second_son in idx[0:2*n_crosses:2], idx{1:2*n_crosses:2], children[0:2*n_crosses:2], children[1:2*n_crosses:2]:
            cross(parent[first_parent_idx], parent[second_parent_idx], first_son, second_son)

        children[2*n_crosses:] = parent[idx[n_crosses:]].copy()

        for son in children[0:2*n_crosses]:
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        n_evals += 2*n_crosses
        if n_evals >= max_evals:
            breakif n_evals >= max_evals:
                break

        #mutation
        mutant_children_idx = np.random.randint(0, p_size, n_mutations)
        mutant_genes_idx = np.random.randint(0, n, n_mutations)

        for son, gen_idx in children[mutant_children_idx], mutant_genes_idx:
            mutate[son["chromosome"], gen_idx]
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        n_evals += n_mutations
        if n_evals >= max_evals:
            break

        #replacement with elitism
        children.sort(order="score")

        if not isSolHere(parent[-1]["chromosome"], children["chromosome"]):
            children[0] = parent[-1]

        parent = children

        parent.sort(order="score")


    return parent[-1] #devolvemos la mejor solución pues se habrá ido manteniendo
