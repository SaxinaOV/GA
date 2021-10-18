import random
import math
import copy
from numpy import random as npr
from numpy import mean

crossover_p = 0.1
mutation_p = 0.1
count_cities = 50
cities = [x for x in range(1, count_cities+1)]

def make_ordered_list(tour):
    cities_list = [x for x in range(1, count_cities+1)]
    ordered_list = []
    for i in range(len(tour)):
        index = cities_list.index(tour[i])
        ordered_list.append(index)
        cities_list.pop(index)
    return ordered_list

def make_tour(ordered_list):
    cities_list = [x for x in range(1, count_cities+1)]
    tour = []
    for i in range(len(ordered_list)):
        index = ordered_list[i]
        tour.append(cities_list[index])
        cities_list.pop(index)
    return tour

def initialization_of_population(size):
    population = []
    for i in range(size):
        tour = random.sample(cities, count_cities)
        print("{}. {}".format(i, tour))
        population.append(make_ordered_list(tour))
    '''
    print("population:")
    for i in range(len(population)):
        print("{}. {}".format(i, population[i]))
    print("\n")
    '''
    return population

def calculate_distance(population, coordinates):
    distance = []
    for tour in population:
        tour_real = [0] + make_tour(tour) + [0]
        i = 0
        dist = 0
        while i < len(tour)-1:
            x1 = coordinates[tour_real[i]][1]
            y1 = coordinates[tour_real[i]][2]
            x2 = coordinates[tour_real[i+1]][1]
            y2 = coordinates[tour_real[i+1]][2]
            dist += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            i += 1
        distance.append(dist)
    return distance

def selection(population, distance):
    pop_after_selection = []
    i = 0
    group_size = 10
    while i < len(population) - (group_size - 1):
        best_tour = population[distance.index(min([distance[j] for j in range(i, i+group_size)]))]
        pop_after_selection += [best_tour]*group_size
        i += group_size
    random.shuffle(pop_after_selection)
    return pop_after_selection

def crossover(population, best_chromosome):
    pop_after_crossover = []
    i = 0
    while i < (len(population) - 1):
        p = random.random()
        first_parent = population[i]
        second_parent = population[i+1]
        point = random.randint(1, len(population[i]) - 1)
        if p < crossover_p:
            first_child = first_parent[:point] + second_parent[point:]
            second_child = second_parent[:point] + first_parent[point:]
            pop_after_crossover += [first_child, second_child]
        else:
            pop_after_crossover += [first_parent, second_parent]
        i += 2
    '''
    print("population after crossover:")
    for i in range(len(pop_after_crossover)):
        print("{}. {}".format(i, make_tour(pop_after_crossover[i])))
    print("\n")
    '''
    return pop_after_crossover    

def mutation(population, best_chromosome):
    pop_after_mutation = []
    for x in population:
        p = random.random()
        slice_size = random.randint(1, len(x)/2)
        if p < mutation_p:# and x!= best_chromosome:
            point = random.randint(0, len(x) - slice_size)
            tour = make_tour(x)
            tour = tour[:point] + tour[point:point+slice_size][::-1] + tour[point+slice_size:]
            x = make_ordered_list(tour)
        pop_after_mutation.append(x)
    '''
    print("population after mutation: ")
    for i in range(len(pop_after_mutation)):
        print("{}. {}".format(i, make_tour(pop_after_mutation[i])))
    print("\n")
    '''
    return pop_after_mutation


'''
def fitness_score(population):
    min_func_value = 1000
    for x in population:
        
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