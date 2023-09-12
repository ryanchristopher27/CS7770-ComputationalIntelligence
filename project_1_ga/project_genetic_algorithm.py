# Evolutionary Algorithms Project
# Genetic Algorithm

# Ryan Christopher


# Imports
from genetic_algorithm import GeneticAlgorithm
from itertools import product
import pandas as pd


def main():
    run_genetic_algorithms()

def run_single_genetic_algorithm():
    generations = 200
    dimensions = 20
    population_size = 100
    initialization_type = "gaussian" # gaussian, uniform
    selection_type = "tournament" # rws, tournament
    fitness_function = "rosenbrock" # rastrigin, spherical, rosenbrock
    crossover_type = "binary_mask" # two_point, binary_mask
    mutation_type = ""
    termination_type = "" # generations, required_average
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
                          termination_type=termination_type,
                          eletism=eletism,
                          eletism_size = eletism_size,
    )

    ga.run_generations()
    ga.print_best_chromosome()
    ga.plot_results()


def run_genetic_algorithms():
    generations = 200
    dimensions = 2
    population_size = 10
    initialization_types = ["gaussian", "uniform"]
    selection_types = ["rws", "tournament"]
    fitness_functions = ['rastrigin']#["rastrigin", "spherical", "rosenbrock"]
    crossover_types = ["two_point", "binary_mask"]
    mutation_types = [""]
    termination_types = ["generations", "required_average"]
    mutation_rate = 0.2
    eletism = True
    eletism_size = 2

    # Use itertools.product to create all combinations
    operators = list(product(initialization_types,
                                selection_types,
                                crossover_types,
                                mutation_types,
                                termination_types,
                                fitness_functions,
                                ))
    
    ga_stats = pd.DataFrame({'init_type': [],
                             'sel_type': [],
                             'cross_type': [],
                             'mut_type': [],
                             'term_type': [],
                             'fitness_func': [],
                             'avg': [],
                             'std': [],
                             'fit_score': []
                             })
    
    print(type(ga_stats))

    # Print the combinations
    for i, operator in enumerate(operators):
        initialization_type = operator[0]
        selection_type = operator[1]
        crossover_type = operator[2]
        mutation_type = operator[3]
        termination_type = operator[4]
        fitness_function = operator[5]

        print(f"------ {i+1}/{len(operators)} ------\n")
        # print(operator)

        ga = GeneticAlgorithm(generations=generations,
                          dimensions=dimensions,
                          population_size=population_size,
                          initialization_type=initialization_type,
                          selection_type=selection_type,
                          fitness_function=fitness_function,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_rate=mutation_rate,
                          termination_type=termination_type,
                          eletism=eletism,
                          eletism_size = eletism_size
        )

        ga.run_generations()
        avg, std, fitness_score = ga.get_ga_statistics()
        # ga.print_best_chromosome()
        # ga.plot_results()

        data = {'init_type': initialization_type,
                   'sel_type': selection_type,
                   'cross_type': crossover_type,
                   'mut_type': mutation_type,
                   'term_type': termination_type,
                   'fitness_func': fitness_function,
                   'avg': avg,
                   'std': std,
                   'fit_score': fitness_score,
                   }
        
        
        ga_stat = pd.Series(data)

        ga_stats.loc[len(ga_stats)] = ga_stat
        # ga_stats = ga_stats.append(ga_stat)
        # ga_stats = pd.concat([ga_stat, ga_stats], ignore_index=True)


    print(ga_stats)

    # Sort values based on average
    ga_stats = ga_stats.sort_values(by=['avg'])

    ga_stats.to_csv('results/ga_stats.csv', index=False)  # Set index=False to exclude row numbers in the output


if __name__=="__main__":
    main()