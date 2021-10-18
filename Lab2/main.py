from funcs import *
import matplotlib
matplotlib.use('TkAgg')


def main():
    n = int(input("Функция: "))
    f = funcs[n][0]
    x = funcs[n][1]
    y = funcs[n][2]
    for c in (0.1, 0.3, 0.5):
        print("crossover prob: {}".format(c))
        answer = 0
        gens = 0
        i = 0
        while i < 10:
            count = 0
            define_func(f, x, y, c)
            population = []
            score = []
            best_chromosome = []
            population = initialization_of_population(100)
            generation = 0
            score = fitness_score(population)
            best_chromosome = population[score.index(min(score))]
            """ print("generation: {}".format(generation))
                print("best chromosome: {}".format(best_chromosome)) 
                print("function value: {}\n".format(score[score.index(min(score))]))  """
            #draw(population)
            while generation < 300 and count < 10:
                generation += 1
                pop_after_selection = selection(population, score)
                pop_after_crossover = arithmetical_crossover(pop_after_selection)###################
                random.shuffle(pop_after_crossover)
                pop_after_crossover[0] = best_chromosome
                pop_after_mutation = random_mutation(
                    pop_after_crossover, best_chromosome)
                population = pop_after_mutation
                population[0] = best_chromosome
                score = fitness_score(pop_after_mutation)
                new_best_chromosome = population[score.index(min(score))]
                if new_best_chromosome == best_chromosome:
                    count += 1
                else:
                    count = 0
                best_chromosome = new_best_chromosome
                """ print("generation: {}".format(generation))
                    print("best chromosome: {}".format(best_chromosome)) 
                    print("function value: {}\n".format(score[score.index(min(score))]))  """
            if score[score.index(min(score))] > 1:
                pass
            else:
                answer += score[score.index(min(score))]
                gens += generation if generation == 300 else generation - 10
                i += 1
        #print("generation: {}".format(gens/10))
        #print("function value: {}\n".format(answer/10))
        draw(population)
    input()


main()
