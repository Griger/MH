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
    p_size = 30
    
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
        #selection by binary tournament of two parent
        selected_parent_idx = np.empty(2, dtype=np.int32)

        for idx in selected_parent_idx:
            idx = np.random.randint(np.random.randint(0,p_size), p_size)

        #cross
        children = np.zeros(2, dtype=datatype)

        cross(parent[selected_parent_idx[0]]["chromosome"], parent[selected_parent_idx[1]]["chromosome"], children[0]["chromosome"], children[1]["chromosome"])

        for son in children:
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        n_evals += 2

        if n_evals >= max_evals:
            breakif n_evals >= max_evals:
                break

        #mutation
        mutant_children_idx = np.random.randint(0, 2, n_mutations)
        mutant_genes_idx = np.random.randint(0, n, n_mutations)

        for son, gen_idx in children[mutant_children_idx], mutant_genes_idx:
            mutate[son["chromosome"], gen_idx]
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        n_evals += n_mutations
        if n_evals >= max_evals:
            break

        #replacement with elitism
        children.sort(order="score")

        if children[1]["score"] > parent[1]["score"]:
            parent[1] = children[1]

            if children[0]["score"] > parent[0]["score"]:
                parent[0] = children[0]
        elif children[1]["score"] > parent[0]["score"]:
            parent[0] = children[1]

        parent.sort(order="score")


    return parent[-1] #devolvemos la mejor solución pues se habrá ido manteniendo
