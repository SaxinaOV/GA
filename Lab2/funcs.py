import random
import math
import numpy as np
from numpy import random as npr
from numpy import mean
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


crossover_p = 0.5
mutation_p = 0.01
dim = 2
x1_range = (-500, 500)
x2_range = (-500, 500)

def define_func(f, x, y, c):
    global x1_range, x2_range, cur_function, crossover_p
    cur_function = f
    x1_range = x
    x2_range = y
    crossover_p = c

def fBran(X, Y):
    a = 1
    b = 5.1/(4 * math.pi**2)
    c = 5/math.pi
    d = 6
    e = 10
    f = 1/(8*math.pi)
    return (a*(Y-b*X**2+c*X-d)**2+e*(1-f)*np.cos(X)+e)

cur_function = fBran

def deJong(X, Y):
    return (X**2 + Y**2)

def rotated(X, Y):
    x = (X, Y)
    s = 0
    c = 0
    for i in range(2):
        for j in range(i):
            c += x[j]
        c = c**2
        s += c
    s = X**2 + (X+Y)**2
    return s

def axisParallel(X, Y):
    return (X**2 + 2* Y**2)

def movedAxis(X, Y):
    return (5 * X**2 + 10 * Y**2)

def Rastrigin(X, Y):
    return (20 + X**2 - 10 * np.cos(2 * np.pi * X) + Y**2 - 10 * np.cos(2 * np.pi * Y))

def Schwefel(X, Y):
    return (-1 * X * np.sin(np.sqrt(abs(X)) - Y * np.sin(np.sqrt(abs(Y)))))

def Easom(X, Y):
    return (-1 * np.cos(X) * np.cos(Y) * np.exp(-1 * ((X - np.pi)**2 + (Y - np.pi)**2)))

def Rosenbrock(X, Y):
    return (100 * (Y - X**2)**2 + (1 - X)**2)
 
def draw(population):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.linspace(x1_range[0], x1_range[1], 100)     
    Y = np.linspace(x2_range[0], x2_range[1], 100) 
    X, Y = np.meshgrid(X, Y)
    Z = cur_function(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
    """ surf = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1,
                            cmap=cm.nipy_spectral, linewidth=0.08,
                            antialiased=False)  """
    xs = []
    ys = []
    zs = []
    for c in population:
        xs.append(c[0])
        ys.append(c[1])
        zs.append(cur_function(c[0], c[1]))
    #ax.scatter(xs, ys, zs, c='black')
    plt.show()

funcs = {1 : (deJong, (-5.12, 5.12), (-5.12, 5.12)),
        2 : (rotated, (-65.536, 65.536),(-65.536, 65.536)),
        3 : (Rastrigin, (-5.12, 5.12), (-5.12, 5.12)),
        4 : (Easom, (-20, 20), (-20, 20)),
        5 : (fBran, (-5, 10), (0, 15)),
        6 : (Rosenbrock, (-2.048, 2.048), (-2.048, 2.048))}

def initialization_of_population(size):
    population = []
    pop = []
    for i in range(size):
        x = random.uniform(x1_range[0], x1_range[1])
        y = random.uniform(x2_range[0], x2_range[1])
        chromosome = [x, y]
        population.append(chromosome)
    return population

def fitness_score(population):
    function = []
    for x in population:
        func_value = cur_function(x[0], x[1])
        function.append(func_value)
    return function

def selection(population, score):
    pop_after_selection = []
    i = 0
    group_size = 2
    while i < len(population) - (group_size - 1):
        p = random.random()
        if p < 0.01:
            best_chromosome = population[score.index(max([score[j] for j in range(i, i+group_size)]))]
        else:
            best_chromosome = population[score.index(min([score[j] for j in range(i, i+group_size)]))]
        pop_after_selection += [best_chromosome]*group_size
        i += group_size
    random.shuffle(pop_after_selection)
    return pop_after_selection

def simple_crossover(population):
    pop_after_crossover = []
    i = 0
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p: # and first_parent != best_chromosome and second_parent != best_chromosome:
            point = random.randint(1, dim-1)
            first_child = first_parent[:point] + second_parent[point:]
            second_child = second_parent[:point] + first_parent[point:]
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover   

def arithmetical_crossover(population):
    pop_after_crossover = []
    i = 0
    w = random.random()
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            first_child = []
            second_child = []
            for j in range(dim):
                first_child.append(w * first_parent[j] + (1 - w) * second_parent[j])
                second_child.append(w * second_parent[j] + (1 - w) * first_parent[j])
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover 

def heuristic_crossover(population):
    pop_after_crossover = []
    i = 0
    w = random.random()
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            first_child = []
            second_child = []
            score = fitness_score([first_parent, second_parent])
            best_parent = first_parent if score.index(max(score)) == 0 else second_parent
            worst_parent = first_parent if best_parent == second_parent else second_parent
            for j in range(dim):
                first_child.append(w * (best_parent[j] - worst_parent[j]) + best_parent[j])
            pop_after_crossover += [first_child, first_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover  

def geometrical_crossover(population):
    pop_after_crossover = []
    i = 0
    w = random.random()
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            first_child = []
            second_child = []
            for j in range(dim):
                first_child.append(w * first_parent[j] * (1 - w) * second_parent[j])
                second_child.append(w * second_parent[j] * (1 - w) * first_parent[j])
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover 

def blend_crossover(population):
    pop_after_crossover = []
    i = 0
    w = random.random()
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            first_child = []
            second_child = []
            for j in range(dim):
                I = max(first_parent[j], second_parent[j]) - min(first_parent[j], second_parent[j])
                a = 0.5 #random.random()
                interval_min = min(first_parent[j], second_parent[j]) - I * a
                interval_max = max(first_parent[j], second_parent[j]) + I * a
                first_child.append(random.uniform(interval_min, interval_max))
            pop_after_crossover += [first_child, first_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover

def sbx_crossover(population):
    pop_after_crossover = []
    i = 0
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            first_child = []
            second_child = []
            for j in range(dim):
                u = npr.uniform()
                n = 5
                if u <= 0.5:
                    b = (2 * u) ** (1 / (n + 1))
                else:
                    b = (1 / (2 * (1 - u))) ** (1 / (n + 1))
                first_child.append(0.5 * ((1 - b) * first_parent[j] + (1 + b) * second_parent[j]))
                second_child.append(0.5 * ((1 + b) * first_parent[j] + (1 - b) * second_parent[j]))
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover

def random_mutation(population, best_chromosome):
    pop_after_mutation = []
    for c in population:
        new_c = c
        p = random.random()
        if p < mutation_p:
            gen = random.randint(0, len(c)-1)
            new_c[gen] = random.uniform(-5, 10) if gen == 0 else random.uniform(0,15)
        pop_after_mutation.append(new_c)
    return pop_after_mutation

