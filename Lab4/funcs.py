from numpy import pi, cos, sin, exp, power
from random import choice, random, randint, sample, shuffle
from math import isnan
from statistics import mean
from graphviz import Graph

def plus(a, b):
    return a + b

def minus(a, b):
    return a - b

def multiplication(a, b):
    return a * b

def division(a, b):
    try:
        c = a / b
    except:
        return 0
    return c

def power(a):
    return a ** 2


x1 = 1
x2 = 2
range_ = [i for i in range(-50, 50)]
X1 = sample(range_, 20)
X2 = sample(range_, 20)
terminals = ('x1', 'x2', pi, -1)
operators = (plus, minus, multiplication, multiplication, cos, cos, exp, power)
un_operators = (cos, exp, power)
bi_operators = (plus, minus, multiplication, division)
population_size = 50
max_tree_size = 10
p_m = 0.3
p_c = 0.3

def func(x1, x2):
    return (-1 * cos(x1) * cos(x2) * exp(-1 * ((x1 - pi)**2 + (x2 - pi)**2)))

#f = lambda x1, x2: (-1 * cos(x1) * cos(x2) * exp(-1 * ((x1 - pi)**2 + (x2 - pi)**2)))

def fun_to_string(f):
    if f == plus:
        return '+'
    if f == minus:
        return '-'
    if f == multiplication:
        return '*'
    if f == division:
        return '/'
    if f == cos:
        return 'cos'
    if f == exp:
        return 'exp'
    if f == power:
        return '^'
    if f == pi:
        return 'pi'
    return str(f)

class Tree:
    def __init__(self, left=None, right=None, symbol=0, number=0):
        self.left = left
        self.right = right
        self.symbol = symbol
        self.number = number

def init_tree_full(max_height, height=0, num=0):
    n = num
    if height < max_height:
        if height < max_height - 1:
            s = choice(operators)
            l, n = init_tree_full(max_height, height+1, num+1)
            if s in bi_operators:
                r, n = init_tree_full(max_height, height+1, n+1)
            else:
                r = None
        else:
            s = choice(terminals)
            l = r = None
        t = Tree(l, r, s, num)
    else:
        t = None
    return t, n

def init_tree_grow(max_height, height=0, num=0, full_height=False):
    n = num
    if height < max_height:
        if height == max_height - 1:
                s = choice(terminals)
                full_height=True
        elif height == 0 or full_height == False:
            s = choice(operators)
        else:
            s = choice(operators + terminals)
        if s in terminals:
            l = r = None
        else:
            l, n, full_height = init_tree_grow(max_height, height+1, n+1, full_height)
            if s in bi_operators:
                r, n, full_height = init_tree_grow(max_height, height+1, n+1, full_height)
            else: 
                r = None
        t = Tree(l, r, s, num)
    else:
        t = None
    return t, n, full_height

def get_node(t, num):
    new_t = None
    if t.number == num:
        new_t = t
    while t.number != num and not new_t:
        if t.left != None and not new_t:
            new_t = get_node(t.left, num)
        if t.right != None and not new_t:
            new_t = get_node(t.right, num)
        if t.left == None and t.right == None or new_t == None:
            return None
    return new_t   

def pick_random_nodes(t1, t2):
    max_n1 = max_height(t1)
    max_n2 = max_height(t2)
    n1 = randint(1, max_n1)
    n2 = randint(1, max_n2)
    node1 = get_node(t1, n1)
    while not node1:
        n1 = randint(1, max_n1)
        node1 = get_node(t1, n1)
    node2 = get_node(t2, 1)
    while not node2:
        n2 = randint(1, max_n2)
        node1 = get_node(t2, n2)
    while (node1.symbol in terminals and node2.symbol in operators) or (node1.symbol in operators and node2.symbol in terminals) or (node1.symbol in un_operators and node2.symbol in bi_operators) or (node2.symbol in un_operators and node2.symbol in bi_operators):
        n1 = randint(1, max_n1)
        n2 = randint(1, max_n2)
        node1 = get_node(t1, n1)
        node2 = get_node(t2, n2)
        while not node1:
            n1 = randint(1, max_n1)
            node1 = get_node(t1, n1)
        node2 = get_node(t2, 1)
        while not node2:
            n2 = randint(1, max_n2)
            node1 = get_node(t2, n2)
    return node1, node2

def tree_crossover(p1, p2, type=0):
    node1, node2 = pick_random_nodes(p1, p2)
    if type == 0:
        temp = node1.symbol
        node1.symbol = node2.symbol
        node2.symbol = temp
    return p1, p2

def tree_mutation(t, type=2):
    max_n = max_height(t)
    r = randint(0, max_n)
    node = get_node(t, r)
    while not node:
        r = randint(0, max_n)
        node = get_node(t, r)
    if type == 0:
        if node.symbol in terminals:
            l = list(terminals)
        elif node.symbol in un_operators:
            l = list(un_operators)
        else:
            l = list(bi_operators)
        l.remove(node.symbol)
        node.symbol = choice(l)
    elif type == 1:
        node.symbol = choice(terminals)
        node.left = node.right = None
    elif type == 2:
        while (not node) or (node.symbol in terminals):
            r = randint(0, max_n)
            node = get_node(t, r) 
        s = node.symbol
        node.left = node.right = None
        node = init_tree_grow(round(max_tree_size/2), 0, node.number)[0]
        node.symbol = s          
    return t

def correct_tree(t):
    if t.left == None and t.right == None:
        return None
    if t.symbol in (exp, cos):
        if random() < 0.5:
            t.right = None
            correct_tree(t.left)
        else: 
            t.left = None
            correct_tree(t.right)
    else:
        correct_tree(t.left)
        correct_tree(t.right)
    return t

def max_height(t, h=-1):
    try: 
        r = t.right
    except:
        r = t[0].right
    if r:
        h = max_height(r, h)
    try: 
        l = t.left
    except:
        l = t[0].left
    if l and l.number > h:
        h = max_height(l, h)
    elif l and l.number < h:
        return h
    else:
        return t.number
    return h

def tree_evaluate(t, x1, x2):
    answer = 0
    try:
        s = t.symbol
    except:
        t = t[0]
    if t.symbol in bi_operators:
        op1 = tree_evaluate(t.left, x1, x2)
        op2 = tree_evaluate(t.right, x1, x2)
        answer = t.symbol(op1, op2)
    elif t.symbol in un_operators:
        if t.left:
            op = tree_evaluate(t.left, x1, x2)
            answer = t.symbol(op)
    else:
        if t.symbol == 'x1':
            return x1
        if t.symbol == 'x2':
            return x2
        return t.symbol 
    return answer  

def initialization(size):
    population = []
    for i in range(size):
        population.append(init_tree_grow(max_tree_size)[0])
    return population

def selection(population, score):
    pop = []
    i = 0
    group_size = 2
    while i < len(population) - (group_size - 1):
        best_chromosome = population[score.index(min([score[j] for j in range(i, i+group_size)]))]
        pop += [best_chromosome]*group_size
        i += group_size
    shuffle(pop)
    return pop

def mutation(population):
    pop = []
    for i in population:
        r = random()
        if r < p_m:
            pop.append(tree_mutation(i))
        else:
            pop.append(i)
    return pop

def crossover(population):
    pop = []
    i = 0
    while i < (len(population) - 1):
        p = random()
        first_parent = population[i]
        second_parent = population[i+1]
        if p < p_c:
            t1, t2 = tree_crossover(first_parent, second_parent)
            pop += [t1, t2]
        else:
            pop += [first_parent, second_parent]
        i += 2
    return pop 

def fitness(tree):
    score = 0
    for x in range(len(X1)):
        x1 = X1[x]
        x2 = X2[x]
        tree_ev = tree_evaluate(tree, x1, x2)
        func_ev = func(x1, x2)
        score += abs(tree_ev - func_ev)
    return score/len(X1)

def fitness_score(population):
    score = []
    best_score = float('inf')
    best_individual = 0
    for i in range(len(population)):
        s = fitness(population[i])
        score.append(s)
        if s < best_score:
            best_score = s
            best_individual = population[i]
    return score, best_score, best_individual


def genetic_algorithm():
    population = initialization(population_size)
    generation = 0
    score, best_score, best_individual = fitness_score(population)
    print("generation: {}".format(generation))
    print("best score: {}".format(best_score))
    draw(best_individual)
    while generation < 20:
        generation += 1
        pop_after_selection = selection(population, score)
        pop_after_crossover = crossover(pop_after_selection)
        pop_after_mutation = mutation(pop_after_crossover)
        population = pop_after_mutation
        score, best_score, best_individual = fitness_score(population)
        mean_list = list(filter(lambda x: not isnan(x) and x != float('inf'), score))
        mean_score = mean(mean_list)
        print(generation)
        print(best_score)
        print("generation: {}".format(generation))
        print("best score: {}".format(best_score))
        print("mean score: {}".format(mean_score))
        draw(best_individual)

'''
def tree_to_dic(t, d={}):
    if t:
        node = (str(t.number), fun_to_string(t.symbol))
        d[node] = []
        if t.left:
            d[node].append(tree_to_dic(t.left, {}))
        if t.right:
            d[node].append(tree_to_dic(t.right, {}))
        if not t.right and not t.left:
            return node
    return d
'''

def tree_to_list(t, L=[]):
    i = 0
    if t:
        node = (str(t.number), fun_to_string(t.symbol))
        if not t.right and not t.left:
            return node
        L.append([])
        L[-1].append(node)
        i = len(L)-1
        if t.left:
            l = tree_to_list(t.left, L)
            if isinstance(l, tuple):
                L[i].append(l)
            if isinstance(l, list):
                L[i].append(l[0])
        if t.right:
            l = tree_to_list(t.right, L)
            if isinstance(l, tuple):
                L[i].append(l)
            if isinstance(l, list):
                L[i].append(l[0])
    if i == 0:
        return L
    return L[i][0]
        

def draw(t):
    g = Graph(format='png')
    l = tree_to_list(t, [])
    for line in l:
        for i in line:
            if i == line[0]:
                g.node(line[0][0], label = line[0][1])
            else:
                g.node(i[0], label = i[1])
                g.edge(line[0][0], i[0])
    g.view()



genetic_algorithm()
input()



'''

def selection(population):
    pop = []
    i = 0
    group_size = 2
    while i < len(population) - group_size:
        best_chromosome = population[score.index(max([score[j] for j in range(i, i+group_size)]))]
        pop += [best_chromosome]
        i += 1
    random.shuffle(pop)
    return pop

def extend_population(population):
    while len(population) < population_size:
        population.append(init_tree_grow(max_tree_size))
    return population
'''
