import random
import math
from numpy import random as npr
from numpy import mean

chromosome_size = 20
lower_limit = -5
upper_limit = 5
crossover_p = 0.9
mutation_p = 0.9

def real_to_chromosome(r):
    x_min = lower_limit
    x_max = upper_limit
    g = ((r - x_min)*(2**chromosome_size - 1))/(x_max - x_min)
    return "{0:b}".format(int(g))

def chromosome_to_real(g):
    x_min = lower_limit
    x_max = upper_limit
    r = (g * (x_max - x_min))/(2**chromosome_size - 1) + x_min
    return r

def initialization_of_population(size):
    population = []
    pop = []
    for i in range(size):
        individual = random.uniform(lower_limit, upper_limit)
        pop.append(individual)
        chromosome = real_to_chromosome(individual)
        while len(chromosome) < chromosome_size:
            chromosome = '0' + chromosome
        population.append(chromosome)
    return population

def fitness_score(population):
    function = []
    for x in population:
        int_ch = int(x, 2)
        real_ch = chromosome_to_real(int_ch)
        func_value = (math.exp(-real_ch) - math.exp(real_ch)) * (math.cos(real_ch)) / (math.exp(real_ch) + math.exp(-real_ch))
        #func_value = -3*real_ch**2+2
        function.append(func_value)
    return function

def selection(population, score):
    pop_after_selection = []
    i = 0
    group_size = 5
    while i < len(population) - (group_size - 1):
        best_chromosome = population[score.index(max([score[j] for j in range(i, i+group_size)]))]
        pop_after_selection += [best_chromosome]*group_size
        i += group_size
    random.shuffle(pop_after_selection)
    return pop_after_selection

def crossover(population):
    pop_after_crossover = []
    i = 0
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < crossover_p:
            point = random.randint(1, len(population[i]) - 1)
            first_child = first_parent[:point] + second_parent[point:]
            second_child = second_parent[:point] + first_parent[point:]
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    return pop_after_crossover    

def mutation(population):
    pop_after_mutation = []
    for x in population:
        p = random.random()
        if p < mutation_p:
            point = random.randint(0, len(x) - 2)
            if x[point] == '0':
                x = x[:point] + '1' + x[point+1:]
        pop_after_mutation.append(x)
    return pop_after_mutation

'''
def fitness_score(population):
    function = []
    score = []
    min_func_value = 1000
    for x in population:
        int_ch = int(x, 2)
        real_ch = chromosome_to_real(int_ch)
        func_value = (math.exp(-real_ch) - math.exp(real_ch)) * (math.cos(real_ch)) / (math.exp(real_ch) + math.exp(-real_ch))
        #func_value = -3*real_ch**2+2
        if func_value < min_func_value:
            min_func_value = func_value
        function.append(func_value)
    normalized_function = function
    if min_func_value < 0:
        normalized_function = list(map(lambda x: x - min_func_value, function))
    for f in normalized_function:
        s = f/sum(normalized_function)
        score.append(s)

    print("function values: ")
    for i in range(len(function)):
        print("{}. {}".format(i, function[i]))
    print("\n")
    print("scores: ")
    for i in range(len(score)):
        print("{}. {}".format(i, score[i]))
    print("\n")

    return score


def selection(population, score):
    pop_after_selection = list(npr.choice(population, size=len(population), p=score))
    print("population after selection:")
    for i in range(len(pop_after_selection)):
        print("{}. {}".format(i, pop_after_selection[i]))
    print("\n")
    return pop_after_selection
'''