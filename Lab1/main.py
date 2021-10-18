from funcs import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

def draw(population):
    p = []
    for i in population:
        p.append(chromosome_to_real(int(i, 2)))
    x = np.linspace(-6, 6, 100)
    y = lambda x: (np.exp(-x) - np.exp(x)) * (np.cos(x)) / (np.exp(x) + np.exp(-x))
    #y = lambda x: -3*x**2+2
    plt.plot(x, y(x))
    plt.scatter(p, list(map(lambda x: y(x), p)))
    plt.grid()
    plt.show()

def main():
    while True:
        tic = time.perf_counter()
        population = []
        score = []
        best_chromosome = ''
        population = initialization_of_population(15)
        generation = 0
        score = fitness_score(population)
        print("generation: {}".format(generation))
        best_chromosome = chromosome_to_real(int(population[score.index(max(score))], 2))
        #draw(population)
        while generation < 30:# and abs(best_chromosome - 3.14895) > 0.0001:
            generation += 1
            pop_after_selection = selection(population, score)
            pop_after_crossover = crossover(pop_after_selection)
            pop_after_mutation = mutation(pop_after_crossover)
            score = fitness_score(pop_after_mutation)
            best_chromosome = chromosome_to_real(int(pop_after_mutation[score.index(max(score))], 2))
            print("generation: {}".format(generation))
            print("best chromosome: {}".format(best_chromosome)) 
            population = pop_after_mutation
            #draw(population) 
            toc = time.perf_counter()
            print(f"time: {(toc - tic):0.4f} seconds\n")
        draw(population)

main()

