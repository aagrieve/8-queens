# AIDAN GRIEVE
## CS441 Artificial Intelligence Programming Assignment 2

# DETAILS
## This program finds a solution to the 8-Queens problem - https://en.wikipedia.org/wiki/Eight_queens_puzzle - using a Genetic Algorithm
####
## The Queens' position is represented in an 8 character string of numbers 0-7
## The index of the string corresponds to the columns of a chessboard
## The value stored at each index corresponds to the row the Queen is in

import numpy as np
import matplotlib.pyplot as mpl
import random as ran

# Helpers
########################################################################################


# takes 2 indexes in a string and swaps the characters of said indexes
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


# traverses the state list in order to compute average and find the largest
# this will be used to create a plot of the GA over iterations
def total_Average_Best_Fitness(states):
    sum = 0
    best_val = 0
    best_indi = None
    for state in states:
        sum += state.fitness
        if state.fitness > best_val:
            best_val = state.fitness
            best_indi = state

    return Data_Set(sum, sum / len(states), best_val, best_indi)


########################################################################################


# Data_Set Class
# Likely unnecessary, but I was sick of iterating lists and wanted a more readable object
########################################################################################


class Data_Set:
    def __init__(self, total, avg, best_val, best_node):
        self.total = total
        self.average = avg
        self.best_val = best_val
        self.best_Indi = best_node


########################################################################################


# individual Class
########################################################################################


class Individual:
    def __init__(self, qString):
        self.state_string = qString
        self.fitness = self.determine_Fitness()

    # determines the number of pairs of queens that are non - attacking in this individual board state
    # this is accomplished by counting the number of attacking queens and subtracting from the ceiling of non-attackers
    # in 8 queens, this ceiling is 28
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


# creates random shuffles of the queen_set
# using the knowledge that in a solution of 8queens no queen will share a row or column
# this significantly lowers the number of possible states
def generate_Initial_Population(pop_size):
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


# using probability, selects parents randomly with weight towards higher fitness scores
def select_Parents(pop, fitness_total):
    probs = []
    # get fitness total
    for i in pop:
        norm_fitness = i.fitness / fitness_total
        probs.append(norm_fitness)

    parents = []
    # select on the roulette wheel
    for i in range(0, len(pop)):
        index = ran.choices(range(len(pop)), weights=probs)[0]
        parents.append(pop[index])

    return parents


########################################################################################


# Crossover
########################################################################################


# picks a random spot in parents to split the strings and swap genes
def crossover(parents):
    children = []
    # used to generalized so n-queens can be solved rather than just 8-queens
    n_value = len(parents[0].state_string)

    while parents:
        curr_length = len(parents)

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

    return children


# DEPR -- REQUIRES CHANGES TO MATCH RETURN OF crossover()
# this should perserve order of parents whereas the basic crossover does not. not currently utilized
def ordered_Crossover(parents):
    children = []
    fitness_total = 0
    while parents:
        curr_length = len(parents)

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


# gets the strings ready for swap_Characters
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


# 5% of the time, swap 2 queen's row # in a state
def mutate(states):
    for state in states:
        roll = ran.random()
        if roll > 0.95:
            swap(state)


########################################################################################


# main
########################################################################################


def main():
    population_size = 100
    num_iterations = 1000
    generation_data = []
    display_graph = True
    display_solution = True
    samples = []
    n_val = 0

    # generate initial population
    # if it returns a list w/ len == 1, it is a solution and we don't need to search
    # otherwise, a list w/ len == 2 will be returned and the algorithm proceeds
    init = generate_Initial_Population(population_size)
    solution = None
    if len(init) == 1:
        solution = init[0]
        print("Solution Found! We got lucky and generated one!")
        print(init[0].state_string)
        print(init[0].convert_To_Board())
    else:
        n_val = len(init[0][0].state_string)
        states = init[0]
        fitness_total = init[1]

        # loop to find solution
        for i in range(0, num_iterations):
            # in order to make this more efficient, selection, crossover, mutation, data_set ought to be combined into 1 traversal of states.
            # as it stands, they are seperated loops, making the time complex significantly worse
            parents = select_Parents(states, fitness_total)
            crossover_list = crossover(parents)

            states = crossover_list
            mutate(states)
            samples.append(ran.choice(states))
            # returns a list of data where [0] = total, [1] = average, [2] = best, [3] = best Individual
            new_data_set = total_Average_Best_Fitness(states)
            generation_data.append(new_data_set)
            fitness_total = new_data_set.total
            if new_data_set.best_val == (n_val * ((n_val - 1) / 2)):
                solution = new_data_set.best_Indi
                break

    if solution != None and display_solution == True:
        print("Solution Found!")
        print(f"It took {len(generation_data)} generations to find!")
        # Commented out in order to make printing cleaner. Uncomment to view a sample from each generation
        # print("Samples from each generation")
        # for index in range(0, len(samples)):
        #     print(f"Generation {index + 1}:")
        #     print(samples[index].state_string)
        #     print(samples[index].convert_To_Board())
        #     print()
        print("Solution:")
        print(solution.state_string)
        print(solution.convert_To_Board())

    if display_graph == True:
        # graph stuff
        # get_Graphing()

        x = 1 + np.arange(len(generation_data))
        averages = []
        bests = []
        for gens in generation_data:
            averages.append(gens.average)
            bests.append(gens.best_val)
        averages = np.array(averages)
        bests = np.array(bests)
        best, avg = mpl.subplots()

        avg.plot(x, averages, linewidth=2.0)
        avg.plot(x, bests, linewidth=2.0)
        avg.set(
            xlim=(0, len(generation_data) + 2),
            xticks=np.arange(1, len(generation_data) + 2),
            ylim=(0, solution.fitness + 1),
            yticks=np.arange(1, solution.fitness + 1),
        )

        mpl.xlabel("Population Generation")
        mpl.ylabel("Average (BLUE) and Best (ORANGE) Fitness")
        mpl.title("8-Queens Genetic Algorithm")
        mpl.show()

    else:
        print("No solution found. Requires more generations or a larger population")


if __name__ == "__main__":
    main()

########################################################################################
