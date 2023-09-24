# Genetic Algorithm Class

# Imports
import os
import random
import numpy as np
import math
from chromosome import Chromosome
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import pylab as pl
from IPython import display
import time


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
                 mutation_rate_individual :float,
                 mutation_rate_genes :float,
                 eletism :bool,
                 eletism_size :int,
                 plot :bool = False,
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
        self.mutation_rate_individual = mutation_rate_individual
        self.mutation_rate_genes = mutation_rate_genes
        self.eletism = eletism
        self.eletism_offset = eletism_size
        self.plot = plot

        if not eletism:
            self.eletism_offset = 0
        # else:
            # self.eletism_offset = 2

        if fitness_function == 'rastrigin' or fitness_function == 'himmelblau':
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
            elif self.fitness_function == "himmelblau":
                chromosome.set_fitness_score(himmelblau_function(chromosome.get_genes()))

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

        elif self.crossover_type == '2_parent_average':

            for i in range(0, required_children):
                p1_index = random.randint(0, len(self.population)-1)
                p2_index = random.randint(0, len(self.population)-1)

                child = Chromosome(self.population[p1_index].get_fitness_score())
                
                avg_genes = []
                for j in range(self.dimensions):
                    avg_genes.append(np.mean([self.population[p1_index].get_genes()[j], self.population[p2_index].get_genes()[j]]))

                child.set_genes(avg_genes)

                population_nextgen.append(child)

        # Only 2 Dimensional
        elif self.crossover_type == 'centroid':

            for i in range(0, required_children):
                p1_index = random.randint(0, len(self.population)-1)
                p2_index = random.randint(0, len(self.population)-1)
                p3_index = random.randint(0, len(self.population)-1)

                centroid_x = (self.population[p1_index].get_genes()[0] +
                              self.population[p2_index].get_genes()[0] +
                              self.population[p3_index].get_genes()[0]) / 3
                
                centroid_y = (self.population[p1_index].get_genes()[1] +
                              self.population[p2_index].get_genes()[1] +
                              self.population[p3_index].get_genes()[1]) / 3
                
                child = Chromosome(self.population[p1_index].get_fitness_score())

                child.set_genes([centroid_x, centroid_y])

                population_nextgen.append(child)

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
            # Check if individual should be mutated
            if random.random() <= self.mutation_rate_individual:
                for j in range(self.dimensions):
                    # Check if gene should be mutated
                    if random.random() <= self.mutation_rate_genes:
                        # Gaussian
                        if self.mutation_type == "gaussian":
                            chromosome.genes[j] = chromosome.genes[j] + np.random.normal(loc=chromosome.genes[j], scale=1, size=1).tolist()[0]
                        
                        # Uniform
                        if self.mutation_type == "uniform":
                            chromosome.genes[j] = chromosome.genes[j] + np.random.uniform(low=chromosome.genes[j] - 2, high=chromosome.genes[j] + 2, size=1).tolist()[0]

                        # Swap
                        if self.mutation_type == "swap" and j < self.dimensions - 1:
                            chromosome.genes[j], chromosome.genes[j+1] = chromosome.genes[j+1], chromosome.genes[j]
                            

            population_nextgen.append(copy.deepcopy(chromosome))
        
        self.population = population_nextgen.copy()


    # Generations
    def run_generations(self) -> None:
        self.best_chromo = []
        self.stats_min = np.zeros(self.generations)
        self.stats_max = np.zeros(self.generations)
        self.stats_avg = np.zeros(self.generations)

        self.population_initialization()
        
        if self.plot:
            fig, ax = self.create_countour_2d_plot()

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

                if self.plot:
                    self.countour_2d_plot(fig, ax, i+1, (i == 0 or i == self.generations-1))
            self.completed_generations = self.generations
        elif self.termination_type == 'convergence':
            i = 0
            min_convergence_generations = 15
            # convergence_threshold = 0.05
            avg_convergence_threshold = self.dimensions * 0.3
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
                    # best_fitness_std = np.std(self.stats_avg[-min_convergence_generations:])
                    avg_fitness_avg = np.mean(self.stats_avg[-min_convergence_generations:])

                    # if best_fitness_std < convergence_threshold:
                    if avg_fitness_avg < avg_convergence_threshold:
                        break

                self.next_generation_selection()
                self.crossover()
                self.mutation()

                if self.plot:
                    self.countour_2d_plot(fig, ax, i+1)

                i += 1
            self.completed_generations = i + 1
        else:
            print("----- Incorrect Termination Type -----")

        if self.plot:
            self.close_contour_2d_plot()

    def create_countour_2d_plot(self) -> ():
        # fig, ax = pl.subplots(nrows = 1, ncols = 1, figsize=(9, 9))
        # return fig, ax
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        return fig, ax

    
    def countour_2d_plot(self, fig, ax, generation, save=False) -> None:
        Bounds = 5
        ContourStepSize = 0.05

        a = np.arange(-Bounds-1, Bounds+1, ContourStepSize)
        b = np.arange(-Bounds-1, Bounds+1, ContourStepSize)
        x, y = np.meshgrid(a, b)
        z = self.fitness_functions(x, y)

        ax.clear()  # Clear the previous plot
        ax.scatter([x.genes[0] for x in self.population], [y.genes[1] for y in self.population], edgecolor='r', alpha=0.8)
        ax.contour(x, y, z, levels=np.logspace(-9, 9, 150), cmap='jet', alpha=0.4)

        # Set limits
        ax.set_xlim(-Bounds-1, Bounds+1)
        ax.set_ylim(-Bounds-1, Bounds+1)

        # Set the title with the iteration value
        ax.set_title(f'Generation {generation}')
        
        # Draw the updated figure
        fig.canvas.draw()

        # Save the first and last figures
        if save == True:
            if generation == 1:
                self.save_contour_figure(fig, generation, "start")
            else:
                self.save_contour_figure(fig, generation, "end")
        
        # Pause for animation
        plt.pause(0.1) 

    def close_contour_2d_plot(self) -> None:
        plt.ioff()  # Disable interactive mode
        plt.show()  # Display the final plot

    def save_contour_figure(self, fig, iteration, name) -> None:
        output_folder = 'results/countours'
        filename = os.path.join(output_folder, f"{self.fitness_function}_{name}_{iteration}.png")
        fig.savefig(filename)

    def plot_stats(self) -> None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))
        # plot the statistics
        # Average Plot
        plt.plot(self.stats_min,'r')
        plt.plot(self.stats_avg,'b')
        plt.plot(self.stats_max,'g')
        plt.ylabel('accuracy')
        plt.xlabel('generations')
        plt.show()

        self.save_stat_figure(fig, 'avg_accuracy')

        # Standard Deviation Plot

        # Save Plots

    def save_stat_figure(self, fig, name) -> None:
        output_folder = 'results/stat_plots'
        filename = os.path.join(output_folder, f"{self.fitness_function}_{name}")
        fig.savefig(filename)

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
    

    def get_completed_generations(self) -> int:
        return self.completed_generations

    # All Fitness Functions
    def fitness_functions(self, x, y):
        if(self.fitness_function == 'rastrigin'):
            z = 10*2 + (x**2 - 10*np.cos(2*math.pi*x)) + (y**2 - 10*np.cos(2*math.pi*y)) # Rastrigin function
        elif(self.fitness_function == 'spherical'):    
            z = x**2 + y**2
        elif(self.fitness_function == 'booth'):
            z = (x + (2*y) - 7)**2 + ((2*x) + y - 5)**2
        elif(self.fitness_function == 'himmelblau'):
            z = ((x**2) + y - 11)**2 + (x + (y**2) - 7)**2 # Himmelblau's function

        return z

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
    score = (genes[0] + (2*genes[1]) - 7)**2 + ((2*genes[0]) + genes[1] - 5)**2
    return score

# Himelblau's Function -5 to 5
def himmelblau_function(genes :[]) -> float:
    score = (genes[0]**2 + genes[1] - 11)**2 + (genes[0] + (genes[1]**2) - 7)**2
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

