# Evolutionary Algorithms Project
# Genetic Algorithm

# Ryan Christopher


# Imports
from genetic_algorithm import GeneticAlgorithm


def main():
    generations = 200
    dimensions = 20
    population_size = 100
    initialization_type = "gaussian" # gaussian, uniform
    selection_type = "tournament" # rws, tournament
    fitness_function = "rosenbrock" # rastrigin, spherical, rosenbrock
    crossover_type = "binary_mask" # two_point, binary_mask
    mutation_type = ""
    mutation_rate = 0.2
    eletism = True
    eletism_size = 2

    ga = GeneticAlgorithm(generations=generations,
                          dimensions=dimensions,
                          population_size=population_size,
                          initialization_type=initialization_type,
                          selection_type=selection_type,
                          fitness_function=fitness_function,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_rate=mutation_rate,
                          eletism=eletism,
                          eletism_size = eletism_size
    )

    ga.run_generations()
    ga.print_best_chromosome()
    ga.plot_results()

if __name__=="__main__":
    main()