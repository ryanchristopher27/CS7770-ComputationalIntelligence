# Evolutionary Algorithms Project
# Genetic Algorithm

# Ryan Christopher


# Imports
import numpy as np
from genetic_algorithm import GeneticAlgorithm
from itertools import product
import pandas as pd
from tqdm import tqdm


def main():
    run_genetic_algorithms(dimensions = 2)
    # run_single_genetic_algorithm()
    # run_dimensionality_experiment()
    # run_mutation_vs_crossover_experiment()

def run_single_genetic_algorithm() -> None:
    generations = 50
    dimensions = 2
    population_size = 20
    initialization_type = "uniform" # gaussian, uniform
    selection_type = "tournament" # rws, tournament
    fitness_function = "rastrigin" # rastrigin, spherical, rosenbrock, booth, himmelblau
    crossover_type = "2_parent_average" # two_point, binary_mask, 2_parent_average, centroid
    mutation_type = "gaussian" # gaussian, uniform, swap
    termination_type = "generations" # generations, convergence
    mutation_rate_individual = 0.5
    mutation_rate_genes = 0.3
    elitism = True
    elitism_size = 2
    selection_size = 10
    plot = True

    ga = GeneticAlgorithm(generations=generations,
                          dimensions=dimensions,
                          population_size=population_size,
                          initialization_type=initialization_type,
                          selection_type=selection_type,
                          fitness_function=fitness_function,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_rate_individual=mutation_rate_individual,
                          mutation_rate_genes=mutation_rate_genes,
                          termination_type=termination_type,
                          elitism=elitism,
                          elitism_size=elitism_size,
                          selection_size=selection_size,
                          plot = plot,
    )

    ga.run_generations()
    ga.print_best_chromosome()
    ga.plot_stats()


def run_genetic_algorithms(dimensions :int) -> None:
    generations = 50
    # dimensions = 2
    population_size = 20
    initialization_types = ["gaussian", "uniform"]
    selection_types = ["rws", "tournament"]
    if dimensions == 2:
        # 2 Dimensions
        fitness_functions = ["rastrigin", "spherical", "booth", "himmelblau"] #["rastrigin", "spherical", "rosenbrock"]
    else:
        # All Dimensio
        fitness_functions = ["rastrigin", "spherical"] #["rastrigin", "spherical", "rosenbrock"]
    crossover_types = ["two_point", "binary_mask", "2_parent_average", "centroid"]
    mutation_types = ["gaussian", "uniform", "swap"]
    termination_types = ["generations", "convergence"]
    mutation_rate_individual = 0.5
    mutation_rate_genes = 0.3
    elitism = True
    elitism_size = 2
    selection_size = 10
    plot = False

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
                             'avg_fit_score': [],
                             'std_fit_score': [],
                             'max_fit_score': [],
                             'min_fit_score': [],
                             'avg_gens': [],
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

        print(f"------ {i+1}/{len(operators)} ------")
        print(operator)
        print("-----------------------------------\n")

        fitness_scores = []

        completed_gens = []

        for i in tqdm(range(100)):
            ga = GeneticAlgorithm(generations=generations,
                            dimensions=dimensions,
                            population_size=population_size,
                            initialization_type=initialization_type,
                            selection_type=selection_type,
                            fitness_function=fitness_function,
                            crossover_type=crossover_type,
                            mutation_type=mutation_type,
                            mutation_rate_individual=mutation_rate_individual,
                            mutation_rate_genes=mutation_rate_genes,
                            termination_type=termination_type,
                            elitism=elitism,
                            elitism_size = elitism_size,
                            selection_size=selection_size,
                            plot = plot
            )

            ga.run_generations()
            avg, std, fitness_score = ga.get_ga_statistics()
            # ga.print_best_chromosome()
            # ga.plot_results()

            fitness_scores.append(fitness_score)
            completed_gens.append(ga.get_completed_generations())


        avg_fit_score = np.mean(fitness_scores)
        std_fit_score = np.std(fitness_scores)
        max_fit_score = max(fitness_scores)
        min_fit_score = min(fitness_scores)

        avg_gens = np.mean(completed_gens)

        data = {'init_type': initialization_type,
                   'sel_type': selection_type,
                   'cross_type': crossover_type,
                   'mut_type': mutation_type,
                   'term_type': termination_type,
                   'fitness_func': fitness_function,
                   'avg_fit_score': avg_fit_score,
                   'std_fit_score': std_fit_score,
                   'max_fit_score': max_fit_score,
                   'min_fit_score': min_fit_score,
                   'avg_gens': avg_gens, 
                   }
        
        
        ga_stat = pd.Series(data)

        ga_stats.loc[len(ga_stats)] = ga_stat
        # ga_stats = ga_stats.append(ga_stat)
        # ga_stats = pd.concat([ga_stat, ga_stats], ignore_index=True)


    print(ga_stats)

    # Sort values based on average
    ga_stats = ga_stats.sort_values(by=['avg_fit_score'])

    ga_stats.to_csv('results/ga_stats.csv', index=False)  # Set index=False to exclude row numbers in the output


def run_dimensionality_experiment() -> None:
    generations = 50
    dimensions = 10000
    population_size = 20
    initialization_type = "uniform" # gaussian, uniform
    selection_type = "tournament" # rws, tournament
    fitness_function = "rastrigin" # rastrigin, spherical, rosenbrock, booth, himmelblau
    crossover_type = "2_parent_average" # two_point, binary_mask, 2_parent_average, centroid
    mutation_type = "gaussian" # gaussian, uniform, swap
    termination_type = "generations" # generations, convergence
    mutation_rate_individual = 0.5
    mutation_rate_genes = 0.3
    elitism = True
    elitism_size = 2
    selection_size = 10
    plot = False

    ga = GeneticAlgorithm(generations=generations,
                          dimensions=dimensions,
                          population_size=population_size,
                          initialization_type=initialization_type,
                          selection_type=selection_type,
                          fitness_function=fitness_function,
                          crossover_type=crossover_type,
                          mutation_type=mutation_type,
                          mutation_rate_individual=mutation_rate_individual,
                          mutation_rate_genes=mutation_rate_genes,
                          termination_type=termination_type,
                          elitism=elitism,
                          elitism_size = elitism_size,
                          selection_size=selection_size,
                          plot = plot,
    )

    ga.run_generations()
    ga.print_best_chromosome()
    ga.plot_stats()

def run_mutation_vs_crossover_experiment() -> None:
    generations = 50
    dimensions = 2
    population_size = 20
    initialization_type = "uniform" # gaussian, uniform
    selection_type = "tournament" # rws, tournament
    fitness_function = "rastrigin" # rastrigin, spherical, rosenbrock, booth, himmelblau
    crossover_type = "2_parent_average" # two_point, binary_mask, 2_parent_average, centroid
    mutation_type = "gaussian" # gaussian, uniform, swap
    termination_type = "generations" # generations, convergence
    # mutation_rate_individual = 0.5
    # mutation_rate_genes = 0.3
    elitism = True
    elitism_size = 2
    # selection_size = 10
    plot = False

    m_rate_individuals = [0.1, 0.3, 0.5]
    m_rate_genes = [0.1, 0.3, 0.5]
    selection_sizes = [14, 10, 6]

    operators = list(product(m_rate_individuals,
                             m_rate_genes,
                             selection_sizes,
                            ))
    
    mut_cross_stats = pd.DataFrame({'m_rate_individual': [],
                             'm_rate_gene': [],
                             'crossover_size': [],
                             'avg_fit_score': [],
                             'std_fit_score': [],
                             'max_fit_score': [],
                             'min_fit_score': [],
                             'avg_gens': [],
                             })
    
    for i, operator in enumerate(operators):
        m_rate_individual = operator[0]
        m_rate_gene = operator[1]
        selection_size = operator[2]

        print(f"------ {i+1}/{len(operators)} ------")
        print(operator)
        print("-----------------------------------\n")

        fitness_scores = []

        completed_gens = []
        for i in tqdm(range(100)):
            ga = GeneticAlgorithm(generations=generations,
                                dimensions=dimensions,
                                population_size=population_size,
                                initialization_type=initialization_type,
                                selection_type=selection_type,
                                fitness_function=fitness_function,
                                crossover_type=crossover_type,
                                mutation_type=mutation_type,
                                mutation_rate_individual=m_rate_individual,
                                mutation_rate_genes=m_rate_gene,
                                termination_type=termination_type,
                                elitism=elitism,
                                elitism_size = elitism_size,
                                selection_size=selection_size,
                                plot = plot,
            )

            ga.run_generations()
            avg, std, fitness_score = ga.get_ga_statistics()

            fitness_scores.append(fitness_score)
            completed_gens.append(ga.get_completed_generations())

            # ga.print_best_chromosome()
            # ga.plot_stats()

        avg_fit_score = np.mean(fitness_scores)
        std_fit_score = np.std(fitness_scores)
        max_fit_score = max(fitness_scores)
        min_fit_score = min(fitness_scores)

        avg_gens = np.mean(completed_gens)

        data = {'m_rate_individual': m_rate_individual,
                   'm_rate_gene': m_rate_gene,
                   'crossover_size': population_size - (selection_size + elitism_size),
                   'avg_fit_score': avg_fit_score,
                   'std_fit_score': std_fit_score,
                   'max_fit_score': max_fit_score,
                   'min_fit_score': min_fit_score,
                   'avg_gens': avg_gens, 
                   }
        
        ga_stat = pd.Series(data)

        mut_cross_stats.loc[len(mut_cross_stats)] = ga_stat

        mut_cross_stats.to_csv('results/mutation_vs_crossover_experiment/m_vs_c_stats.csv', index=False)  # Set index=False to exclude row numbers in the output


        

if __name__=="__main__":
    main()