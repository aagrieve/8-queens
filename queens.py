# AIDAN GRIEVE
## CS441 Artificial Intelligence Programming Assignment 2

# DETAILS
## This program finds a solution to the 8-Queens problem - https://en.wikipedia.org/wiki/Eight_queens_puzzle - using a Genetic Algorithm
####
## The Queens' position is represented in an 8 character string of numbers 0-7
## The index of the string corresponds to the columns of a chessboard
## The value stored at each index corresponds to the row the Queen is in

import numpy as np
import math as m
import random as ran

# Helpers
########################################################################################


def swap_Characters(string, index1, index2):
    if index1 < 0 or index2 < 0 or index1 >= len(string) or index2 >= len(string):
        raise IndexError("Out of range")

    # get chars we want swapped
    char1 = string[index1]
    char2 = string[index2]

    # do this horrible splitting of strings because strings are immutable :(
    b4_char1 = string[:index1]
    between_char1_and_char2 = string[index1 + 1 : index2]
    if index2 > len(string) - 1:
        after_char2 = ""
    else:
        after_char2 = string[index2 + 1 :]

    return b4_char1 + char2 + between_char1_and_char2 + char1 + after_char2


########################################################################################


# individual Class
########################################################################################


class Individual:
    def __init__(self, qString):
        self.state_string = qString
        self.fitness = self.determine_Fitness()

    # determines the number of pairs of queens that are attacking in this individual board state
    def determine_Fitness(self):
        board = self.convert_To_Board()
        length = len(self.state_string)
        count = 0
        fitness = int(length * (length - 1) / 2)
        for i in range(0, length - 1):
            curr = int(self.state_string[i])
            # row
            slice = board[curr, i:]
            freq = np.count_nonzero(slice)
            if freq > 1:
                count += freq - 1

            # column
            slice = board[curr:, i]
            freq = np.count_nonzero(slice)
            if freq > 1:
                count += freq - 1

            # diag down
            slice = []
            i_copy = i
            j = curr
            while j < 8 and i_copy < 8:
                slice.append(board[j, i_copy])
                j += 1
                i_copy += 1
            freq = np.count_nonzero(slice)
            if freq > 1:
                count += freq - 1

            # diag up
            slice = []
            i_copy = i
            j = curr
            while j >= 0 and i_copy < 8:
                slice.append(board[j, i_copy])
                j -= 1
                i_copy += 1
            freq = np.count_nonzero(slice)
            if freq > 1:
                count += freq - 1
        return fitness - count

    # converts an 8 character string into a 8x8 numpy array so that fitness can be determined and board can be displayed
    def convert_To_Board(self):
        length = len(self.state_string)
        board_state = np.zeros((length, length))
        for i in range(0, length):
            board_state[int(self.state_string[i]), i] = 1
        return board_state


########################################################################################


# Population Functions
########################################################################################


def generate_Initial_Population(pop_size):
    # this may need to be completely random
    queen_set = {0, 1, 2, 3, 4, 5, 6, 7}
    pop_list = []
    fitness_total = 0

    for i in range(0, pop_size):
        copy = list(queen_set.copy())
        ran.shuffle(copy)
        copy = "".join(map(str, copy))
        new_indiv = Individual(copy)
        if new_indiv.fitness == 28:
            return [new_indiv]
        fitness_total += new_indiv.fitness
        pop_list.append(new_indiv)

    return pop_list, fitness_total


########################################################################################


# Parents
########################################################################################


def select_Parents(pop, fitness_total):
    probs = []
    # get fitness total
    for i in pop:
        norm_fitness = i.fitness / fitness_total
        probs.append(norm_fitness)

    parents = []
    # select on the roulette wheel
    for i in range(0, len(pop)):
        # this doesnt fix but doesn't really do anything differently from old version. looks cleaner tho
        index = ran.choices(range(len(pop)), weights=probs)[0]
        parents.append(pop[index])

    return parents


########################################################################################


# Crossover
########################################################################################
def crossover(parents):
    children = []
    fitness_total = 0

    while parents:
        curr_length = len(parents)
        # print(curr_length)
        # remove curr parents from list
        parent1 = parents.pop(ran.randint(0, curr_length - 1))
        parent2 = parents.pop(ran.randint(0, curr_length - 2))

        # child1
        snip = ran.randint(0, len(parent1.state_string) - 1)

        child1 = parent1.state_string[:snip]

        for gene in parent2.state_string:
            if gene not in child1:
                child1 = child1 + gene

        # child2
        snip = ran.randint(0, len(parent2.state_string) - 1)

        child2 = parent2.state_string[:snip]

        for gene in parent2.state_string:
            if gene not in child2:
                child2 = child2 + gene

        children.append(Individual(child1))
        children.append(Individual(child2))
        if children[-1].fitness == 28:
            return [children[-1]]
        if children[-2].fitness == 28:
            return [children[-2]]
        fitness_total += children[-1].fitness + children[-2].fitness

    return (children, fitness_total)


# something is wrong here -- MIGHT BE FINE -- TEST ONCE YOU FIX PARENT SELECTION
# the children are not being changed enough
def ordered_crossover(parents):
    children = []
    fitness_total = 0
    while parents:
        curr_length = len(parents)
        # print(curr_length)
        # remove curr parents from list
        parent1 = parents.pop(ran.randint(0, curr_length - 1))
        parent2 = parents.pop(ran.randint(0, curr_length - 2))

        # get random segment to snip for child1
        snip_start = ran.randint(0, len(parent1.state_string) - 2)
        snip_end = ran.randint(snip_start + 1, len(parent1.state_string) - 1)
        if snip_start > snip_end:
            temp = snip_start
            snip_start = snip_end
            snip_end = temp
        child1 = (
            parent1.state_string[:snip_start]
            + parent1.state_string[snip_start:snip_end]
        )

        for gene in parent2.state_string:
            if gene not in child1:
                child1 = child1 + gene

        # repeat for child2
        snip_start = ran.randint(0, len(parent1.state_string) - 2)
        snip_end = ran.randint(snip_start + 1, len(parent1.state_string) - 1)
        if snip_start > snip_end:
            temp = snip_start
            snip_start = snip_end
            snip_end = temp
        child2 = (
            parent2.state_string[:snip_start]
            + parent2.state_string[snip_start:snip_end]
        )

        for gene in parent1.state_string:
            if gene not in child2:
                child2 = child2 + gene

        # append to children list
        children.append(Individual(child1))
        children.append(Individual(child2))
        if children[-1].fitness == 28:
            return [children[-1]]
        if children[-2].fitness == 28:
            return [children[-2]]
        fitness_total += children[-1].fitness + children[-2].fitness

    return (children, fitness_total)


########################################################################################


# Mutation
########################################################################################


def swap(state):
    keep_looping = True
    swap_start = 0
    swap_end = 0

    # loop if the random nums are equal
    while keep_looping:
        swap_start = ran.randint(0, len(state.state_string) - 1)
        swap_end = ran.randint(0, len(state.state_string) - 1)
        if swap_start != swap_end:
            keep_looping = False

    # get the smaller value into start
    if swap_start > swap_end:
        temp = swap_start
        swap_start = swap_end
        swap_end = temp

    state.state_string = swap_Characters(state.state_string, swap_start, swap_end)


def mutate(states):
    for state in states:
        roll = ran.random()
        if roll > 0.95:
            swap(state)


########################################################################################


# main
########################################################################################


def main():
    population_size = 250
    num_iterations = 1000

    # generate initial population
    init = generate_Initial_Population(population_size)
    if len(init) == 1:
        print("Solution Found! We got lucky and generated one!")
        print(init[0].state_string)
        print(init[0].convert_To_Board())
    else:
        states = init[0]
        # test
        # for i in states:
        #     print(i.state_string)
        # print()
        fitness_total = init[1]
        # loop to find solution
        for i in range(0, num_iterations):
            # print(fitness_total)
            parents = select_Parents(states, fitness_total)
            # test
            # for j in parents:
            #     print(j.state_string)
            crossover_list = crossover(parents)
            if len(crossover_list) == 1:
                print("Solution Found!")
                print(f"It took {i} generations to find!")
                print(crossover_list[0].state_string)
                print(crossover_list[0].convert_To_Board())
                break
            states = crossover_list[0]
            mutate(states)
            fitness_total = crossover_list[1]


if __name__ == "__main__":
    main()

########################################################################################
