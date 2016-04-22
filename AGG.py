import numpy as np
from knnLooGPU import *

def cross(first_parent, second_parent, a,b):
    second_parent_gens

def mutate(s, gen_idx):
    s[gen_idx] = not s[gen_idx]

def AGG(train_data, train_labels, knnGPU):
    #poblacion inicial
    n = len(train_data[0])
    p_size = 30 #poblation size

    P0 = np.random.choice([True, False], n*p_size).reshape((p_size, n)) #initial poblation
