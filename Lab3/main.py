from funcs import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import combinations, chain


def draw(tour, coordinates):
    x = [coordinates[tour[i]][1] for i in range(len(tour))]
    y = [coordinates[tour[i]][2] for i in range(len(tour))]
    plt.plot(x, y)
    plt.scatter(x, y, color='r')
    plt.scatter(coordinates[tour[-2]][1], coordinates[tour[-2]][2], color='g')
    plt.scatter(coordinates[0][1], coordinates[0][2], color='g')
    plt.grid(True)
    plt.show()

def main():
    
    while True:
        tic = time.perf_counter()
        coordinates = []
        cities_file = open('cities.txt', 'r')
        for line in cities_file:
            l = line.split(' ')
            ints = [int(i) for i in l]
            coordinates.append(ints)
        cities_file.close()
        '''
        answer = []
        f = open('best.txt', 'r')
        for line in f:
            answer.append(int(line)-1)
        f.close()
        answer.append(0)
        i = 0
        dist = 0
        while i < len(answer)-1:
            x1 = coordinates[answer[i]][1]
            y1 = coordinates[answer[i]][2]
            x2 = coordinates[answer[i+1]][1]
            y2 = coordinates[answer[i+1]][2]
            dist += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            i += 1
        print(dist)
        draw(answer, coordinates)
        '''
        population = []
        distance = []
        population = initialization_of_population(700)
        generation = 0
        min_distance = 100000
        min_index = 0
        count = 0
        prev_best_tour = 0
        minimal_distance = 10000
        minimal_tour = 0
        while generation < 700 and count < 50:
            distance = calculate_distance(population, coordinates)
            population = selection(population, distance)
            generation += 1
            min_distance = 100000
            min_index = 0
            for i in range(len(distance)):
                if distance[i] < min_distance:
                    min_index = i
                    min_distance = distance[i]
                if distance[i] < minimal_distance:
                    minimal_distance = distance[i]
                    minimal_tour = population[min_index] 
            best_tour = make_tour(population[min_index])
            best_chromosome = population[min_index]
            print("generation: {}".format(generation))
            print("minimal distance: {}".format(min_distance))
            print("best tour: {}".format(best_tour))
            if best_tour == prev_best_tour:
                count += 1
            else:
                count = 0
            pop_after_crossover = crossover(population, best_chromosome)
            pop_after_mutation = mutation(pop_after_crossover, best_chromosome)
            population = pop_after_mutation
            population[0] = best_chromosome
            prev_best_tour = best_tour
            if generation % 10 == 0 or generation == 1:
                draw([0] + best_tour + [0], coordinates)
        print("generation: {}".format(generation))
        print("minimal distance: {}".format(min_distance))
        print("minimal distance of all generations: {}".format(minimal_distance))
        toc = time.perf_counter()
        print(f"time: {(toc - tic):0.4f} seconds\n")
        draw([0] + best_tour + [0], coordinates)
        #print("minimal distance: {}".format(min_distance))
        #print("best tour: {}".format(best_tour, cities.copy()))

main()
'''
coordinates = []
cities_file = open('cities.txt', 'r')
for line in cities_file:
    l = line.split(' ')
    ints = [int(i) for i in l]
    coordinates.append(ints)
cities_file.close()

answer = []
f = open('my_best.txt', 'r')
for line in f:
    answer.append(int(line)-1)
f.close()
answer.append(0)
i = 0
dist = 0
while i < len(answer)-1:
    x1 = coordinates[answer[i]][1]
    y1 = coordinates[answer[i]][2]
    x2 = coordinates[answer[i+1]][1]
    y2 = coordinates[answer[i+1]][2]
    dist += math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    i += 1
print(dist)
draw(answer, coordinates)


nodes = [(1,4), (3.5,5), (6,4), (3.5,3)]

combs = list(combinations(nodes, 2))

x = list(chain(*[[p1[0],p2[0]] for p1,p2 in combs]))
y = list(chain(*[[p1[1],p2[1]] for p1,p2 in combs]))

plt.plot(x, y, marker='o')

x2 = [1, 3.5, 6, 3.5, 1]
y2 = [4, 5, 4, 3, 4]
plt.plot(x2, y2, color="red")
plt.show()
'''

