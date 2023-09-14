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

    completed_generations = 0
    
    mu = 0

    def __init__(self, 
                 generations :int, 
                 dimensions :int, 
                 population_size :int, 
                 initialization_type :str, 
                 selection_type :str, 
                 fitness_function :str, 
                 crossover_type :str,
                 mutation_type :str,
                 termination_type :str,
                 mutation_rate :float,
                 eletism :bool,
                 eletism_size :int,
                ) -> None:
        self.generations = generations # number of generations
        self.dimensions = dimensions # number of features
        self.population_size = population_size # number of individuals
        self.initialization_type = initialization_type # type of initialization
        self.selection_type = selection_type
        self.fitness_function = fitness_function
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.termination_type = termination_type # termination criteria
        self.mutation_rate = mutation_rate
        self.eletism = eletism
        self.eletism_offset = eletism_size

        if not eletism:
            self.eletism_offset = 0
        # else:
            # self.eletism_offset = 2

        if fitness_function == 'rastrigin' or fitness_function == 'himelblaus':
            self.sigma = 1.7
            self.max = 5
        elif fitness_function == 'spherical' or fitness_function == 'rosenbrock' or fitness_function == 'booth':
            self.sigma = 3.3
            self.max = 10


    # population initialization
    def population_initialization(self) -> None:
        population = []

        if self.initialization_type == "gaussian":
            for i in range(self.population_size):
                chromosome = Chromosome(None)
                chromosome.set_genes(np.random.normal(loc=self.mu, scale=self.sigma, size=self.dimensions).tolist())
                # chromosome.set_genes([random.gauss(mu, sigma) for x in range(self.dimensions)])
                # chromosome = [random.gauss(mu, sigma) for x in range(self.dimensions)]
                population.append(chromosome)
        elif self.initialization_type == "uniform":
            # scores = np.random.uniform(low=-5.12, high=5.12, size=self.population_size).tolist()
            for i in range(self.population_size):
                chromosome = Chromosome(None)
                chromosome.set_genes(np.random.uniform(low=(-1)*(self.max), high=self.max, size=self.dimensions).tolist())
                population.append(chromosome)

        self.population = population.copy()


    # fitness score - updates the fitness score within all the population chromosomes
        # sorts the population from highest score to lowest
    def fitness_score(self) -> None:
        for chromosome in self.population:
            # Set fitness score within each chromosome
            # chromosome.set_fitness_score(rastrigin_function(chromosome.get_genes(), self.dimensions))
            if self.fitness_function == "rastrigin":
                chromosome.set_fitness_score(rastrigin_function(chromosome.get_genes(), self.dimensions))
            elif self.fitness_function == "spherical":
                chromosome.set_fitness_score(sphere_function(chromosome.get_genes()))
            elif self.fitness_function == "rosenbrock":
                chromosome.set_fitness_score(rosenbrock_function(chromosome.get_genes(), self.dimensions))
            elif self.fitness_function == "booth":
                chromosome.set_fitness_score(booth_function(chromosome.get_genes()))
            elif self.fitness_function == "himelblaus":
                chromosome.set_fitness_score(himelblaus_function(chromosome.get_genes()))

        # Sort population based on fitness score
        self.population.sort(key=lambda x: x.fitness_score) 
            
    # generation selection
    def next_generation_selection(self) -> None:
        population_nextgen = []
        # Include eletism
        if self.eletism:
            for i in range(self.eletism_offset):
                population_nextgen.append(copy.deepcopy(self.population[i]))
            # population_nextgen.append(copy.deepcopy(self.population[1]))

        if self.selection_type == "rws":
            # Implement RWS
            selections = roulette_wheel_selection(self.population[self.eletism_offset:], int((self.population_size - self.eletism_offset)/2))
        elif self.selection_type == "tournament":
            # Implement Tournament Selection
            selections = tournament_selection(self.population[self.eletism_offset:], int((self.population_size - self.eletism_offset)/2), self.population_size - self.eletism_offset)
        
        population_nextgen += selections

        self.population = population_nextgen.copy()


    # crossover
    def crossover(self) -> None:
        population_nextgen = []

        required_children = self.population_size - len(self.population)

        # if self.eletism:
        #     population_nextgen.append(copy.deepcopy(self.population[0]))
        #     population_nextgen.append(copy.deepcopy(self.population[1]))

        if self.crossover_type == 'two_point':
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

        elif self.crossover_type == 'binary_mask':
            # Binary Mask Crossover

            for i in range(0, required_children, 2):
                child_1 = copy.deepcopy(self.population[i])
                child_2 = copy.deepcopy(self.population[i+1])

                binary_mask = [random.randint(0, 1) for x in range(self.dimensions)]

                for j in range(self.dimensions):
                    if binary_mask[j] == 1:
                        # Swap the values
                        child_1.genes[j], child_2.genes[j] = child_2.genes[j], child_1.genes[j]

                population_nextgen.append(child_1)
                population_nextgen.append(child_2)

        self.population += population_nextgen

        # self.population = population_nextgen.copy()


    # mutation
    def mutation(self) -> None:
        population_nextgen = []
        
        # Elitism
        for j in range(self.eletism_offset):
            population_nextgen.append(copy.deepcopy(self.population[j]))
        # population_nextgen.append(copy.deepcopy(self.population[0]))
        # population_nextgen.append(copy.deepcopy(self.population[1]))

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

        if self.termination_type == 'generations':
            # for i in tqdm(range(self.generations)):
            for i in range(self.generations):
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
            self.completed_generations = self.generations
        elif self.termination_type == 'convergence':
            i = 0
            min_convergence_generations = 5
            convergence_threshold = 0.05
            while (True):
                # Evaluate Fitness and Order Chromosomes
                self.fitness_score()

                self.best_chromo.append(copy.deepcopy(self.population[0]))

                scores = [x.fitness_score for x in self.population]

                self.stats_min[i] = np.min(scores)
                self.stats_max[i] = np.amax(scores)
                self.stats_avg[i] = np.mean(scores)

                # Check if convergence is met
                if i > min_convergence_generations:
                    best_fitness_std = np.std(self.stats_min[-min_convergence_generations:])

                    if best_fitness_std < convergence_threshold:
                        break

                self.next_generation_selection()
                self.crossover()
                self.mutation()

                i += 1
            self.completed_generations = i + 1
        else:
            print("----- Incorrect Termination Type -----")

    
    def plot_results(self) -> None:
        # plot the statistics
        plt.plot(self.stats_min,'r')
        plt.plot(self.stats_avg,'b')
        plt.plot(self.stats_max,'g')
        plt.ylabel('accuracy')
        plt.xlabel('generations')
        plt.show()

    def print_best_chromosome(self) -> None:
        print("-- FIRST GEN --")
        print("Chromo: ", self.best_chromo[0].get_genes())
        print("Score: ", self.best_chromo[0].get_fitness_score())
        print("-- FINAL GEN --")
        print("Chromo", self.best_chromo[-1].get_genes())
        print("Score: ", self.best_chromo[-1].get_fitness_score())

    def get_ga_statistics(self) -> ():
        genes = self.best_chromo[-1].get_genes()
        fitness_score = self.best_chromo[-1].get_fitness_score()

        avg = np.average(genes)
        std = np.std(genes)

        return avg, std, fitness_score
    
    def get_plot(self) -> ():
        # Average Plot
        plt.plot(self.stats_min,'r')
        plt.plot(self.stats_avg,'b')
        plt.plot(self.stats_max,'g')
        plt.ylabel('accuracy')
        plt.xlabel('generations')
        avg_plot = plt

        # Standard Deviation Plot


        # Values Plot

    def get_completed_generations(self) -> int:
        return self.completed_generations



# ----------------------------------------------------------------------------------
### Helper Functions

## Test Fitness Functions

### ------ Multi Dimensional -----------
# Rastrigin Function
def rastrigin_function(genes :[], dimensions :int) -> float:
    A = 10
    score = (A*dimensions) + sum([(x**2 - A * np.cos(2 * math.pi * x)) for x in genes])
    return score
# Spherical Function
def sphere_function(genes :[]) -> float:
    score = sum([x**2 for x in genes])
    return score

### ----- > 2 Dimensional ------------
# Rosenbrock Function
def rosenbrock_function(genes :[], dimensions :int) -> float:
    score = sum([(100 * ((genes[i+1] - (genes[i]**2))**2) + (1 - genes[i])**2) for i in range(len(genes)-2)])
    return score

### ------ 2 Dimensional -------------
# Booth Function -10 to 10
def booth_function(genes :[]) -> float:
    score = (genes[0] + 2*genes[1] + 7)**2 + (2 * genes[0] + genes[1] - 5)**2
    return score

# Himelblau's Function -5 to 5
def himelblaus_function(genes :[]) -> float:
    score = (genes[0]**2 + genes[1] + 11)**2 + (genes[0] + genes[1]**2 - 7)**2
    return score


# Roulette Wheel Selection
# Gives higher priority to lower fitness score
def roulette_wheel_selection(chromosomes :[], num_to_select :int) -> []:
    selected_chromosomes = []
    scores = [x.fitness_score for x in chromosomes]
    min_score = np.min(scores)
    max_val = np.sum([min_score/score for score in scores])
    for i in range(num_to_select):
        pick = random.uniform(0, max_val)
        current = 0
        for j in range(len(chromosomes)): 
            current += (min_score / chromosomes[j].get_fitness_score())
            if current > pick: 
                selected_chromosomes.append(copy.deepcopy(chromosomes[j])) 
                break 
        # selected_genes.append(copy.deepcopy(genes[j])) 

    return selected_chromosomes


# Tournament Selection
def tournament_selection(chromosomes :[], num_to_select :int, population_size: int) -> []:
    selected_chromosomes = []
    nt_size = 5
    for i in range(num_to_select):
        indices = [random.randint(0, population_size-1) for x in range(nt_size)]

        # find tournament winner between indices
        chromosomes_subset = [chromosomes[j] for j in indices]
        chromosomes_subset.sort(key=lambda x: x.fitness_score) 
        # add tournament winner to selected genes
        selected_chromosomes.append(copy.deepcopy(chromosomes_subset[0]))

    return selected_chromosomes

