# Genetic Algorithm Class

# Imports
import random
import numpy as np
import math
from chromosome import Chromosome
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy

class GeneticAlgorithm:

    population = []

    best_chromo = []
    stats_min = []
    stats_max = []
    stats_avg = []

    def __init__(self, 
                 generations :int, 
                 dimensions :int, 
                 population_size :int, 
                 initialization_type :str, 
                 selection_type :str, 
                 fitness_function :str, 
                 crossover_type :str,
                 mutation_type :str,
                 mutation_rate :float,
                 eletism :bool
                ) -> None:
        self.generations = generations # number of generations
        self.dimensions = dimensions # number of features
        self.population_size = population_size # number of individuals
        self.initialization_type = initialization_type # type of initialization
        self.selection_type = selection_type
        self.fitness_function = fitness_function
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.mutation_rate = mutation_rate
        self.eletism = eletism

        if eletism:
            self.eletism_offset = 2
        else:
            self.eletism_offset = 0


    # population initialization
    def population_initialization(self) -> None:
        population = []
        mu = 0
        sigma = 3
        for i in range(self.population_size):
            chromosome = Chromosome(None)
            chromosome.set_genes([random.gauss(mu, sigma) for x in range(self.dimensions)])
            # chromosome = [random.gauss(mu, sigma) for x in range(self.dimensions)]
            population.append(chromosome)
        self.population = population.copy()


    # fitness score - updates the fitness score within all the population chromosomes
        # sorts the population from highest score to lowest
    def fitness_score(self) -> None:
        for chromosome in self.population:
            # Set fitness score within each chromosome
            chromosome.set_fitness_score(rastrigin_function(chromosome.get_genes(), self.dimensions))
        # Sort population based on fitness score
        self.population.sort(key=lambda x: x.fitness_score) 
            
    # generation selection
    def next_generation_selection(self) -> None:
        population_nextgen = []
        # Include eletism
        if self.eletism:
            population_nextgen.append(copy.deepcopy(self.population[0]))
            population_nextgen.append(copy.deepcopy(self.population[1]))

        # Implement RWS
        selections = roulette_wheel_selection(self.population[self.eletism_offset:], int((self.population_size - self.eletism_offset)/2))
        population_nextgen += selections

        self.population = population_nextgen.copy()


    # crossover
    def crossover(self) -> None:
        population_nextgen = []

        required_children = self.population_size - len(self.population)

        # if self.eletism:
        #     population_nextgen.append(copy.deepcopy(self.population[0]))
        #     population_nextgen.append(copy.deepcopy(self.population[1]))

        # Two Point Crossover
        # for i in range(self.eletism_offset, self.population_size, 2):
        for i in range(0, required_children, 2):
            child_1 = copy.deepcopy(self.population[i])
            child_2 = copy.deepcopy(self.population[i+1])

            first_cross_point = random.randint(0,self.dimensions)
            second_cross_point = random.randint(0,self.dimensions)
            # did we get the same point? have to deal with that
            if( first_cross_point == second_cross_point ):
                first_cross_point = 0
                second_cross_point = self.dimensions
            # are our swap indices not in order? have to deal with that
            if( first_cross_point > second_cross_point ):
                swaper = first_cross_point
                first_cross_point = second_cross_point
                second_cross_point = swaper

            # Swap
            child_1.genes[first_cross_point:second_cross_point] = self.population[i].genes[first_cross_point:second_cross_point]
            child_2.genes[first_cross_point:second_cross_point] = self.population[i+1].genes[first_cross_point:second_cross_point]

            population_nextgen.append(child_1)
            population_nextgen.append(child_2)

        self.population += population_nextgen

        # self.population = population_nextgen.copy()


    # mutation
    def mutation(self) -> None:
        population_nextgen = []
        
        # Elitism
        population_nextgen.append(copy.deepcopy(self.population[0]))
        population_nextgen.append(copy.deepcopy(self.population[1]))

        for i in range(self.eletism_offset, self.population_size):
            chromosome = copy.deepcopy(self.population[i])
            for j in range(self.dimensions):
                random_val = random.random()
                if random_val < self.mutation_rate/2:
                    chromosome.genes[j] = chromosome.genes[j] - random.random()
                elif random_val > self.mutation_rate/2:
                    chromosome.genes[j] = chromosome.genes[j] + random.random()
            population_nextgen.append(copy.deepcopy(chromosome))
        
        self.population = population_nextgen.copy()


    # Generations
    def run_generations(self) -> None:
        self.best_chromo = []
        self.stats_min = np.zeros(self.generations)
        self.stats_max = np.zeros(self.generations)
        self.stats_avg = np.zeros(self.generations)

        self.population_initialization()

        for i in tqdm(range(self.generations)):
            # Evaluate Fitness and Order Chromosomes
            self.fitness_score()

            self.best_chromo.append(copy.deepcopy(self.population[0]))

            scores = [x.fitness_score for x in self.population]

            self.stats_min[i] = np.min(scores)
            self.stats_max[i] = np.amax(scores)
            self.stats_avg[i] = np.mean(scores)

            self.next_generation_selection()
            self.crossover()
            self.mutation()

    
    def plot_results(self) -> None:
        # plot the statistics
        plt.plot(self.stats_min,'r')
        plt.plot(self.stats_avg,'b')
        plt.plot(self.stats_max,'g')
        plt.ylabel('accuracy')
        plt.xlabel('generations')
        plt.show()



# ----------------------------------------------------------------------------------

### Helper Functions

# Rastrigin Function
def rastrigin_function(genes :[], dimensions :int) -> float:
    A = 10
    score = (A*dimensions) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in genes])
    return score

# Roulette Wheel Selection
def roulette_wheel_selection(genes :[], num_to_select :int) -> []:
        selected_genes = []
        scores = [x.fitness_score for x in genes]
        min_score = np.min(scores)
        max_val = np.sum([min_score/score for score in scores])
        for i in range(num_to_select):
            pick = random.uniform(0, max_val)
            current = 0
            for j in range(len(genes)): 
                current += (min_score / genes[j].get_fitness_score())
                if current > pick: 
                    selected_genes.append(copy.deepcopy(genes[j])) 
                    break 
            # selected_genes.append(copy.deepcopy(genes[j])) 

        return selected_genes