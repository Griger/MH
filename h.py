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

n = 10
p_size = 6
size_chromosome_string = str(n) + 'bool'
datatype = np.dtype( [('chromosome',size_chromosome_string), ('score',np.float32)] )

parent = np.zeros(p_size, dtype=datatype)
parent["chromosome"] = np.random.choice([True,False], (p_size, n))

selected_parent_idx = np.empty(p_size, dtype=np.int32)

for idx in range(0,p_size):
    selected_parent_idx[idx] = np.random.randint(np.random.randint(0,p_size), p_size)

selected_pairs = zip(selected_parent_idx[0::2], selected_parent_idx[1::2])

print selected_pairs
children = np.zeros(p_size, dtype=datatype)


for p_pair, first_son, second_son in zip(selected_pairs, range(0,p_size,2), range(1,p_size,2)):
    print first_son
    print second_son
    a,b = cross(parent[p_pair[0]]["chromosome"], parent[p_pair[1]]["chromosome"])
    children[first_son] = a
    children[second_son] = b

print(children)
