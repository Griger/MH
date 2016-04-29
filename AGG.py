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
    #poblacion inicial
    n = len(train_data[0])
    p_size = 30 #poblation size

    P0 = np.random.choice([True, False], n*p_size).reshape((p_size, n)) #initial poblation
