import numpy as np
from knnLooGPU import *
from math import *

def cross(first_parent, second_parent, first_son, second_son):
    eq_idx = first_parent == second_parent
    neq_idx = np.logical_not(eq_idx)

    #copy identical genes into children
    first_son = first_parent & eq_idx
    second_son = first_parent & eq_idx

    #random different genes selection
    neq_size = sum(neq_idx)
    first_son[neq_idx] = np.random.choice([True, False], neq_size)
    second_son[neq_idx] = np.logical_not(first_son[neq_idx])

def mutate(s, gen_idx):
    s[gen_idx] = not s[gen_idx]

def isSolHere(sol, sols_set):
    equal_elements = sol == sols_set
    return len(sol) in equal_elements.sum(axis = 1)

def AGG(train_data, train_labels, knnGPU):
    max_evals = 300
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
        n_evals += 1
        individual["score"] = knnGPU.scoreSolution(train_data[:,individual["chromosome"]], train_labels)

    parent.sort(order="score")

    while (n_evals < max_evals):
        #selection by binary tournament
        selected_parent_idx = np.empty(p_size, dtype=np.int32)

        for idx in range(0,p_size):
            selected_parent_idx[idx] = np.random.randint(np.random.randint(0,p_size), p_size)

        selected_pairs = zip(selected_parent_idx[0::2], selected_parent_idx[1::2])
        print(selected_pairs)

        #cross
        children = np.zeros(p_size, dtype=datatype)

        for p_pair, first_son, second_son in zip(selected_pairs, children[0::2], children[1::2]):
            cross(parent[p_pair[0]]["chromosome"], parent[p_pair[1]]["chromosome"], first_son["chromosome"], second_son["chromosome"])

        children[2*n_crosses:] = parent[selected_parent_idx[2*n_crosses:]].copy()
        print("hijos", children)

        for son in children[0:2*n_crosses]:
            n_evals += 1
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        print("scores", children["scores"])
        if n_evals >= max_evals:
            break

        #mutation
        mutant_children_idx = np.random.randint(0, p_size, n_mutations)
        mutant_genes_idx = np.random.randint(0, n, n_mutations)

        for son, gen_idx in zip(children[mutant_children_idx], mutant_genes_idx):
            n_evals += 1
            mutate(son["chromosome"], gen_idx)
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        if n_evals >= max_evals:
            break

        #replacement with elitism
        children.sort(order="score")

        if not isSolHere(parent[-1]["chromosome"], children["chromosome"]):
            children[0] = parent[-1]

        parent = children

        parent.sort(order="score")


    print("Se han hecho ", n_evals, " llamadas a la función objetivo.\n")
    return parent[-1] #devolvemos la mejor solución pues se habrá ido manteniendo
