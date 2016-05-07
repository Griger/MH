import numpy as np

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

def h(s):
    print "Me han pasado: \n", s

n = 10
p_size = 6 #population size

cross_p = 0.7
n_crosses = 2

mutation_p = 0.001
n_mutations = 5

n_evals = 0

size_chromosome_string = str(n) + 'bool'
datatype = np.dtype( [('chromosome',size_chromosome_string), ('score',np.float32)] )

parent = np.zeros(p_size, dtype=datatype)
parent["chromosome"] = np.random.choice([True,False], (p_size, n)) #random initial population

for individual in parent:
    n_evals += 1
    individual["score"] = np.random.randint(0,100)

parent.sort(order="score")

print "Los padres tras generarlos, darles score y ordenados \n", parent
#selection by binary tournament
selected_parent_idx = np.empty(p_size, dtype=np.int32)

for idx in range(0,p_size):
    selected_parent_idx[idx] = np.random.randint(np.random.randint(0,p_size), p_size)

print "Padres seleccionados por torneo binario:", selected_parent_idx
selected_pairs = zip(selected_parent_idx[0:2*n_crosses:2], selected_parent_idx[1:2*n_crosses:2])

#cross
children = np.zeros(p_size, dtype=datatype)

for p_pair, idx1, idx2 in zip(selected_pairs, range(0,2*n_crosses,2), range(1,2*n_crosses,2)):
    print p_pair
    children["chromosome"][idx1], children["chromosome"][idx2] = cross(parent["chromosome"][p_pair[0]], parent["chromosome"][p_pair[1]])

children[2*n_crosses:] = parent[selected_parent_idx[2*n_crosses:]].copy()

for son in children[0:2*n_crosses]:
    n_evals += 1
    son["score"] = np.random.randint(0,100)

print "Hijos tras el cruce\n", children

#mutation
mutant_children_idx = np.random.randint(0, p_size, n_mutations)
mutant_genes_idx = np.random.randint(0, n, n_mutations)

print "Las mutaciones"
print mutant_children_idx
print mutant_genes_idx

for idx, gen_idx in zip(mutant_children_idx, mutant_genes_idx):
    n_evals += 1
    mutate(children["chromosome"][idx], gen_idx)
    children["score"][idx] = np.random.randint(0,100)

print "Hijos tras mutacion \n", children
#replacement with elitism
children.sort(order="score")

print "El mejor hijo tiene score: ", children["score"][-1]
print "El mejor padre tiene score: ", parent["score"][-1]

if children["score"][-1] < parent["score"][-1]:
    children[0] = parent[-1]

print "El peor hijo es ahora: \n", children[0]

parent = children

print(parent)
