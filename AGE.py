import numpy as np
from knnLooGPU import *
from math import *

def cross(first_parent, second_parent):
    eq_idx = first_parent == second_parent
    neq_idx = np.logical_not(eq_idx)

    #copy identical genes into children
    first_son = first_parent & eq_idx
    second_son = first_parent & eq_idx

    #random different genes selection
    neq_size = sum(neq_idx)
    first_son[neq_idx] = np.random.choice([True, False], neq_size)
    second_son[neq_idx] = np.logical_not(first_son[neq_idx])
    return first_son, second_son

def mutate(s, gen_idx):
    s[gen_idx] = not s[gen_idx]

def AGE(train_data, train_labels, knnGPU):
    max_evals = 15000
    n = len(train_data[0])
    p_size = 30

    mutation_p = 0.001
    mutations_threshold = 2 * n /1000.0

    n_evals = 0

    size_chromosome_string = str(n) + 'bool'
    datatype = np.dtype( [('chromosome',size_chromosome_string), ('score',np.float32)] )

    parent = np.zeros(p_size, dtype=datatype)
    parent["chromosome"] = np.random.choice([True,False], (p_size, n)) #random initial population

    for individual in parent:
        n_evals += 1
        individual["score"] = knnGPU.scoreSolution(train_data[:,individual["chromosome"]], train_labels)

    parent.sort(order="score")

    while (n_evals < max_evals):
        #selection by binary tournament of two parent
        idx1 = np.random.randint(np.random.randint(0,p_size), p_size)
        idx2 = np.random.randint(np.random.randint(0,p_size), p_size)

        #cross
        children = np.zeros(2, dtype=datatype)

        children["chromosome"][0], children["chromosome"][1] = cross(parent["chromosome"][idx1], parent["chromosome"][idx2])

        for son in children:
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        n_evals += 2

        #mutation
        if np.random.random() <= mutations_threshold:
            mutant_children_idx = np.random.randint(0, 2)
            mutant_gen_idx = np.random.randint(0, n)
            mutate(children["chromosome"][mutant_children_idx], mutant_gen_idx)
            children["score"][mutant_children_idx] = knnGPU.scoreSolution(train_data[:,children["chromosome"][mutant_children_idx]], train_labels)
            n_evals += 1

        #replacement with elitism
        children.sort(order="score")

        if children["score"][1] > parent["score"][1]:
            parent[1] = children[1]

            if children["score"][0] > parent["score"][0]:
                parent[0] = children[0]
        elif children["score"][1] > parent["score"][0]:
            parent[0] = children[1]

        parent.sort(order="score")


    return parent[-1] #devolvemos la mejor solución pues se habrá ido manteniendo
