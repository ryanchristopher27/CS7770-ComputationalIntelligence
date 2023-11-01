# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from zadeh_rule import Zadeh_Rule
from zadeh_inference import Zadeh_Inference
from zadeh_fuzzy_set import Zadeh_Fuzzy_Set


class Zadeh_FIS:
    # membership_functions = {}

    def __init__(self):
        self.domains = {}
        self.fuzzy_sets = {} # Fuzzy Sets that represent linguistic variables
        self.rules = {}
        self.implication_operator = "Correlation_Min"
        self.inferences = {}

    def set_implication_operator(self, operator :str) -> None:
        self.implication_operator = operator

    # Domains
    # ----------------------------------------------------------------------
    def create_domain(self, name: str, domain :[]) -> None:
        self.domains[name] = domain
    

    # Membership Functions
    # ----------------------------------------------------------------------
    def create_fuzzy_set(self, name :str, domain_involved :str, fuzzy_set :[]) -> None:
        # Check to make sure it fits domain
        if len(fuzzy_set) != len(self.domains[domain_involved]):
            print(f'Fuzzy Set {fuzzy_set} does not fit Domain {domain_involved}')

        self.fuzzy_sets[name] = Zadeh_Fuzzy_Set(name, domain_involved, fuzzy_set)


    # Rules
    # ----------------------------------------------------------------------
    def create_rule(self, name :str, antecedents :[str], consequent :str) -> None:
        self.rules[name] = Zadeh_Rule(name, antecedents, consequent)

        # Step 1 - Combine Antecedents
        antecedent_matrix, vector_lengths = self.combine_antecedents(antecedents)
        # Step 2 - Cylindrical Projection
        consequent_set = self.fuzzy_sets[consequent].get_fuzzy_set()
        relational_matrix = self.cylindrical_projection(consequent_set, antecedent_matrix, vector_lengths)

        # Set Relation
        print(relational_matrix)
        self.rules[name].set_relation(relational_matrix)
        
    
    # Step 1
    def combine_antecedents(self, antecedents :[]) -> ():
        num_antecedents = len(antecedents)
        if num_antecedents == 1:
            return self.fuzzy_sets[antecedents[0]].get_fuzzy_set(), [len(self.fuzzy_sets[antecedents[0]].get_fuzzy_set())]
        else:
            lengths = []
            for i in range(len(antecedents)):
                lengths.append(len(self.fuzzy_sets[antecedents[i]].get_fuzzy_set()))
            antecedent_matrix = np.zeros(lengths, dtype=float)

            if num_antecedents == 2:
                for i, ant_i in enumerate(self.fuzzy_sets[antecedents[0]].get_fuzzy_set()):
                    for j, ant_j in enumerate(self.fuzzy_sets[antecedents[1]].get_fuzzy_set()):
                        antecedent_matrix[i][j] = self.perform_implication_operator([ant_i, ant_j], 1, "Combine")
                        # antecedent_matrix[i][j] = min(ant_i, ant_j)
            elif num_antecedents == 3:
                for i, ant_i in enumerate(self.fuzzy_sets[antecedents[0]].get_fuzzy_set()):
                    for j, ant_j in enumerate(self.fuzzy_sets[antecedents[1]].get_fuzzy_set()):
                        for k, ant_k in enumerate(self.fuzzy_sets[antecedents[2]].get_fuzzy_set()):
                            antecedent_matrix[i][j][k] = self.perform_implication_operator([ant_i, ant_j, ant_k], 1, "Combine")
                            # antecedent_matrix[i][j][k] = min(ant_i, ant_j, ant_k)
        
            return antecedent_matrix, lengths
        
    # Step 2 - Cylindrical Projection onto consequent
    def cylindrical_projection(self, consequent :[], antecedent_matrix :[], vector_lengths :[]) -> []:
        vector_lengths.append(len(consequent))
        projected_matrix = np.zeros(vector_lengths, dtype=float)

        if len(vector_lengths) - 1 == 1:
            for i, ant in enumerate(antecedent_matrix):
                for j, cons in enumerate(consequent):
                    projected_matrix[i][j] = self.perform_implication_operator([ant], cons, "Cyl_Proj")
                    # projected_matrix[i][j] = min(ant, cons)
        elif len(vector_lengths) - 1 == 2:
            for i in range(vector_lengths[0]):
                for j in range(vector_lengths[1]):
                    for k, cons in enumerate(consequent):
                        projected_matrix[i][j][k] = self.perform_implication_operator([antecedent_matrix[i][j]], cons, "Cyl_Proj")
                        # projected_matrix[i][j][k] = min(antecedent_matrix[i][j], cons)
        elif len(vector_lengths) - 1 == 3:
            for i in range(vector_lengths[0]):
                for j in (vector_lengths[1]):
                    for k in (vector_lengths[2]):
                        for l, cons in enumerate(consequent):
                            projected_matrix[i][j][k][l] = self.perform_implication_operator([antecedent_matrix[i][j][k]], cons, "Cyl_Proj")
                            # projected_matrix[i][j][k][l] = min(antecedent_matrix[i][j][k], cons)

        return projected_matrix

    
    # Evalutation
    # ----------------------------------------------------------------------
    def evaluate_zadeh(self, rule :str, antecedents :[]) -> {}:
        num_antecedents = len(antecedents)

        relational_matrix = self.rules[rule].get_relation()
        antecedent_matrix, lengths = self.combine_antecedents(antecedents)

        inferred_matrix = np.zeros(relational_matrix.shape, dtype=float)

        output_set = np.zeros(len(self.fuzzy_sets[self.rules[rule].get_consequent()].get_fuzzy_set()), dtype=float)
        try:
            if num_antecedents == 1:
                for i, row in enumerate(relational_matrix):
                    for j, val in enumerate(row):
                        inferred_matrix[i][j] = np.minimum(val, antecedent_matrix[i])

                for k, col in enumerate(zip(*inferred_matrix)):
                    output_set[k] = max(col)

            # CHECK SUPPORT FOR 2 AND 3 ANTECEDENTS
            #///////////////////////////////////////////////////////////////////////////
            elif num_antecedents == 2:
                for i, matrix in enumerate(relational_matrix):
                    for j, row in enumerate(matrix):
                        for k, val in enumerate(row):
                            inferred_matrix[i][j][k] = np.minimum(val, antecedent_matrix[i][j])

                for l, plane in enumerate(zip(*inferred_matrix)):
                    output_set[l] = max(plane)
                
            elif num_antecedents == 3:
                for i, matrix_3d in enumerate(relational_matrix):
                    for j, matrix in enumerate(matrix_3d):
                        for k, row in enumerate(matrix):
                            for l, val in enumerate(row):
                                inferred_matrix[i][j][k][l] = np.minimum(val, antecedent_matrix[i][j][k])

                for m, cube in enumerate(zip(*inferred_matrix)):
                    output_set[m] = max(cube)
            #///////////////////////////////////////////////////////////////////////////
        except:
            print("Antecedent Order / Sizes are not congruent with relational matrix")

        consequent_domain = self.fuzzy_sets[self.rules[rule].get_consequent()].get_domain()
        # Create new Zadeh Inference for consequent domain
        if consequent_domain not in self.inferences:
            self.inferences[consequent_domain] = Zadeh_Inference(consequent_domain=consequent_domain)
        
        # Add Output Domain to Zadeh Inference instance
        self.inferences[consequent_domain].add_output_set(name=rule, output_set=output_set)

        print(output_set)

    # Defuzzification
    # ----------------------------------------------------------------------
    def defuzzify(self, domain :str, aggregation_method :str) -> ():
        zadeh_inference = self.inferences[domain]

        aggregated_set = np.zeros(len(self.domains[domain]))
        for _, fuzzy_set in zadeh_inference.get_output_sets().items():
            for i in range(len(self.domains[domain])):
                # Aggregate sets
                if aggregation_method == 'Sum':
                    aggregated_set[i] = aggregated_set[i] + fuzzy_set[i]
                elif aggregation_method == 'Max':
                    aggregated_set[i] = np.maximum(aggregated_set[i], fuzzy_set[i])
                else:
                    print(f'{aggregation_method} is not supported.')
                    exit(0)

        output = self.domains[domain][pd.Series(aggregated_set).idxmax()]
        relation_val = max(aggregated_set)

        return output, relation_val

    # Helpers
    # ----------------------------------------------------------------------
    def perform_implication_operator(self, antecedents :[], consequent :float, step :str) -> float:
        if self.implication_operator == "Lukasiewicz":
            if step == "Combine":
                result = min(antecedents)
            else:
                result = np.minimum(1, (1.0 - sum(antecedents) + consequent))
        elif self.implication_operator == "Correlation_Min":
            if step == "Combine":
                result = max(antecedents)
            else:
                result = np.minimum(antecedents, consequent)
        elif self.implication_operator == "Correlation_Product":
            if step == "Combine":
                result = np.prod(antecedents)
            else:
                result = np.prod(antecedents) * consequent
        else:
            print(f'{self.implication_operator} Implication Operator is not supported.')
            exit(0)

        return result
