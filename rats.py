from PIL import Image, ImageOps
import sys
import numpy as np
import random
import collections
from functools import reduce
from multiprocessing import Pool, Queue

#global
MAX_MOVES = 5000
POPULATION_SIZE = 500
MUTATION_FACTOR = 5
NUM_SELECTED = 20


class MyOD(collections.OrderedDict):
    def update_pos (self, pos, val):
        for i, k in enumerate(self):
            if i == pos:
                self[k] += val
                return

#GeneticRat defined with Memory Array and Rules Dictionary
class GeneticRat:
    def __init__(self, number, rules, memory=None, pos=None):

        self.rules = MyOD(rules)
        size = (21, 21)
        self.memory = np.zeros(size)
        self.x =1
        self.y =0
        self.number = number

    def getMemory(self):
        return self.memory

    def setMemory(self, posx, posy):
        #this is specifically to update the memory incrementing by 1 everytime the rat moves
        self.memory[posx][posy] += 1

    def printMemory(self):
        print(self.memory)

    def getRules(self):
        return self.rules

    def setRules(self, d):
        self.rules = d

    def setRule(self, key, value):
        self.rules[key] = value    

    def printRules(self):
        print(self.rules)

    def getPos(self):
        return (self.x, self.y)

    def setPos(self, x, y):
        self.x = x
        self.y = y

    def printPos(self):
        print((self.x,self.y))

#Crossover operation takes two Parent rules dicts and outputs a new mixed child rule dict
def crossover(d1, d2):
    r_idx = random.randint(0, 15)

    child_rules = []

    items1 = list(d1.items())
    items2 = list(d2.items())

    for idx in range(0,r_idx) :
        child_rules.append(items1[idx])
    for idx in range(r_idx, 16):
        child_rules.append(items2[idx])
    child_rules = MyOD(child_rules)

    return child_rules

#Add a random amount of mutation to all of the weights
def mutate(d):
    i = random.randint(0,(len(d)))
    d.update_pos(i, random.random() * MUTATION_FACTOR * random.randint(-1,1))

    return d

#Define bfs to find the shortest route through the maze from a point
#This will form the fitness function to select rats from a generatin to breed together
def bfs(grid, start):
    wall, clear, goal = 1, 0, 2
    width, height = 21, 21
    queue = collections.deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if grid[y][x] == goal:
            return len(path)
        for x2, y2 in ((x+1,y), (x-1,y), (x, y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and grid[y2][x2] != wall and (x2, y2) not in seen:
                queue.append(path + [(x2,y2)])
                seen.add((x2,y2))

def run_maze(data):
    num_moves = 0
    np_img = data[0]
    theRat = data[1]
    while (num_moves < MAX_MOVES) and (theRat.getPos() != (19,20)):
        #the rat remembers where it has been
        theRat.setMemory(theRat.x, theRat.y)

        #set the rat's choices
        choices = ['', '', '', '']
        if np_img[theRat.x][theRat.y-1] == 1:
            choices[0] = 'Nx'
        elif theRat.y-1 < 0:
            choices[0] = 'Nx'
        elif theRat.getMemory()[theRat.x][theRat.y-1] == 1:
            choices[0] = 'N1'
        elif theRat.getMemory()[theRat.x][theRat.y-1] > 1:
            choices[0] = 'N+'
        else:
            choices[0] = 'N0'

        if np_img[theRat.x][theRat.y+1] ==1:
            choices[1] = 'Sx'
        elif theRat.y+1 > 19:
            choices[1] = 'Sx'
        elif theRat.getMemory()[theRat.x][theRat.y+1] == 1:
            choices[1] = 'S1'
        elif theRat.getMemory()[theRat.x][theRat.y+1] > 1:
            choices[1] = 'S+'
        else:
            choices[1] = 'S0'

        if np_img[theRat.x+1][theRat.y] == 1:
            choices[2] = 'Ex'
        elif theRat.x + 1 > 19:
            choices[2] = 'Ex'
        elif theRat.getMemory()[theRat.x+1][theRat.y] == 1:
            choices[2] = 'E1'
        elif theRat.getMemory()[theRat.x+1][theRat.y] > 1:
            choices[2] = 'E+'
        else:
            choices[2] = 'E0'

        if np_img[theRat.x-1][theRat.y] == 1:
            choices[3] = 'Wx'
        elif theRat.x - 1 < 0:
            choices[3] = 'Wx'
        elif theRat.getMemory()[theRat.x-1][theRat.y] == 1:
            choices[3] = 'W1'
        elif theRat.getMemory()[theRat.x-1][theRat.y] > 1:
            choices[3] = 'W+'
        else:
            choices[3] = 'W0'

        rand = random.random()

        normalized_choices = [0,0,0,0]
        sum_choices = 0
        for choice in choices:
            sum_choices += theRat.rules[choice]

        for i in range(len(choices)):
            normalized_choices[i] = theRat.rules[choices[i]]/sum_choices

        if rand < normalized_choices[0]:
            if choices[0] == 'Nx':
                theRat.y += 0
            else:
                theRat.y -= 1
        elif rand < normalized_choices[0] + normalized_choices[1]:
            if choices[1] == 'Sx':
                theRat.y += 0
            else:
                theRat.y += 1
        elif rand < normalized_choices[0] + normalized_choices[1] + normalized_choices[2]:
                if choices[2] == 'Ex':
                    theRat.x += 0
                else:
                    theRat.x += 1
        else:
            if choices[3] == 'Wx':
                theRat.x += 0
            else:
                theRat.x -= 1

        num_moves += 1

    dist_from_goal = bfs(np_img, (theRat.x, theRat.y))
    score = num_moves + dist_from_goal
    return num_moves, dist_from_goal, score

                           
def average(lst):
    return reduce(lambda a, b: a+b, lst) / len(lst)


def select(rats, scores):
    selected = []
    fitness = [x[2] for x in scores]
    for i in range(len(fitness)):
        data = rats[i].getMemory()
        redundancy = np.sum(np.where(data > 2))
        fitness += redundancy * 5
    results = list(zip(rats, fitness))
    while len(selected) < NUM_SELECTED:
        selected.append(min(results, key = lambda t:t[1]))
        results.remove(min(results, key = lambda t:t[1]))

    return selected

def breed(selected):
    parents = [x[0] for x in selected]
    children = []
    c_idx = 0
    while len(children) < POPULATION_SIZE:
        p_idx1= random.randint(0, 9)
        p_idx2 = random.randint(0,9)
        child_rules = crossover(parents[p_idx1].getRules(), parents[p_idx2].getRules())
        child_rules = mutate(child_rules)
        children.append(GeneticRat(c_idx, rules = child_rules))
        c_idx += 1
    return children

def main():
    #create the initial generation

    rules = [('Nx', 1.0), ('Sx', 1.0), ('Ex',1.0), ('Wx',1.0),
                      ('N0',1.0), ('S0',1.0), ('E0',1.0), ('W0',1.0),
                      ('N1',1.0), ('S1',1.0), ('E1',1.0), ('W1',1.0),
                      ('N+',1.0), ('S+',1.0), ('E+',1.0), ('W+',1.0)]
    g_idx = 0
    current_generation = []
    continue_generation = True

    #processQueue = Queue()
    #outputQueue = Queue()

    for i in range(POPULATION_SIZE):
        current_generation.append(GeneticRat(i,rules))

    while continue_generation:
        img = Image.open("1.png").convert('L')
        img_inverted = ImageOps.invert(img)

        np_img = np.array(img_inverted)
        np_img[np_img > 0] = 1

        np_img[19][20] = 2

        scores = []

        for i in range(0, len(current_generation)):
            current_generation[i] = (np_img, current_generation[i])

        #for all of my mazes the start is np_img[1][0] and the end is np_img[19][20]
#        for rat in current_generation:
#            processQueue.put(rat)

        with Pool(processes=4) as pool:
            scores = pool.map(run_maze, current_generation)

#        while not outputQueue.empty():
 #           data = outputQueue.get()
 #           scores[data[0]] = data[1]
 
        print("Generation: " + str(g_idx))
        print("Average number of moves: " + str(int(average([x[0] for x in scores]))))
        print("Average distance from solution: " + str(int(average([x[1] for x in scores]))))
        g_idx += 1

        for i in range(0, len(current_generation)):
            current_generation[i] = current_generation[i][1]

        selected = select(current_generation,  scores)
        current_generation = breed(selected)

if __name__ == '__main__':
    main()
