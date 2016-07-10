import numpy as np
import BLCUDA1iter as BL
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

def AM(train_data, train_labels, knnGPU, model):
    print("Ejecutando AGG Hibrido con modelo",model)
    max_evals = 15000
    n = len(train_data[0])
    p_size = 10 #population size

    cross_p = 0.7
    n_crosses = ceil(p_size/2 * 0.7)

    n_generations = 0 #counter to apply LS
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
        n_generations += 1
        #selection by binary tournament
        selected_parent_idx = np.empty(p_size, dtype=np.int32)

        for idx in range(0,p_size):
            selected_parent_idx[idx] = np.random.randint(np.random.randint(0,p_size), p_size)

        selected_pairs = zip(selected_parent_idx[0:2*n_crosses:2], selected_parent_idx[1:2*n_crosses:2])

        #cross
        children = np.zeros(p_size, dtype=datatype)

        for p_pair, idx1, idx2 in zip(selected_pairs, range(0,2*n_crosses,2), range(1,2*n_crosses,2)):
            children["chromosome"][idx1], children["chromosome"][idx2] = cross(parent["chromosome"][p_pair[0]], parent["chromosome"][p_pair[1]])

        children[2*n_crosses:] = parent[selected_parent_idx[2*n_crosses:]].copy()


        for son in children[0:2*n_crosses]:
            n_evals += 1
            son["score"] = knnGPU.scoreSolution(train_data[:,son["chromosome"]], train_labels)

        #mutation
        mutant_children_idx = np.random.randint(0, p_size, n_mutations)
        mutant_genes_idx = np.random.randint(0, n, n_mutations)

        for idx, gen_idx in zip(mutant_children_idx, mutant_genes_idx):
            n_evals += 1
            mutate(children["chromosome"][idx], gen_idx)
            children["score"][idx] = knnGPU.scoreSolution(train_data[:,children["chromosome"][idx]], train_labels)

        #replacement with elitism
        children.sort(order="score")

        if children["score"][-1] < parent["score"][-1]:
            children[0] = parent[-1]

        parent = children
        parent.sort(order="score")

        if (n_generations == 10):
            n_generations = 0
            if model == 1:#apply LS over any individual
                for individual in parent:
                    individual["chromosome"], individual["score"], n_evals_in_BL = BL.BLCUDA1iter(train_data, train_labels, knnGPU, individual["chromosome"])
                    n_evals += n_evals_in_BL
            elif model == 2:#apply LS over a random individual
                idx = np.random.randint(0, p_size)
                parent["chromosome"][idx], parent["score"][idx], n_evals_in_BL = BL.BLCUDA1iter(train_data, train_labels, knnGPU, parent["chromosome"][idx])
                n_evals += n_evals_in_BL
            elif model == 3:#apply LS over the best
                parent["chromosome"][-1], parent["score"][-1], n_evals_in_BL = BL.BLCUDA1iter(train_data, train_labels, knnGPU, parent["chromosome"][-1])
                n_evals += n_evals_in_BL

    return parent["chromosome"][-1], parent["score"][-1] #devolvemos la mejor solución pues se habrá ido manteniendo
