# Evolutionary Algorithms Project
# Genetic Algorithm

# Ryan Christopher


# Imports
from genetic_algorithm import GeneticAlgorithm


def main():
    generations = 100
    dimensions = 10
    population_size = 50
    initialization_type = "guassian"
    selection_type = "rws"
    fitness_function = "rastrigin"
    crossover_type = ""
    mutation_type = ""
    mutation_rate = 0.1
    eletism = True

    ga = GeneticAlgorithm(generations=generations,
                          dimensions=dimensions,
                          population_size=population_size,
                          initialization_type=initialization_type,
                          selection_type=selection_type,
                          fitness_function=fitness_function,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_rate=mutation_rate,
                          eletism=eletism
    )

    ga.run_generations()
    ga.plot_results()

if __name__=="__main__":
    main()