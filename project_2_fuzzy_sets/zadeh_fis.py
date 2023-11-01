# Imports
import numpy as np
import matplotlib.pyplot as plt
from zadeh_rule import Zadeh_Rule

class Zadeh_FIS:
    # membership_functions = {}

    def __init__(self):
        self.domains = {}
        self.fuzzy_sets = {} # Fuzzy Sets that represent linguistic variables
        self.rules = {}
        self.implication_operator = "Correlation_Min"

    def set_implication_operator(self, operator :str) -> None:
        self.implication_operator = operator

    # Domains
    # ----------------------------------------------------------------------
    def create_domain(self, name: str, domain :[]) -> None:
        self.domains[name] = domain
    

    # Membership Functions
    # ----------------------------------------------------------------------
    def create_fuzzy_set(self, name :str, domain_involved :str, set :[]) -> None:
        # Check to make sure it fits domain
        if len(set) != len(self.domains[domain_involved]):
            print(f'Fuzzy Set {set} does not fit Domain {domain_involved}')

        self.fuzzy_sets[name] = set


    # Rules
    # ----------------------------------------------------------------------
    def create_rule(self, name :str, antecedents :[str], consequent :str) -> None:
        self.rules[name] = Zadeh_Rule(name, antecedents, consequent)
        
    # Evalutation
    # ----------------------------------------------------------------------
    def evaluate_zadeh(self, rule :str, antecedents :[]) -> {}:
        # Step 1 - Create Relations between antecedents
        antecedent_matrix, vector_lengths = self.combine_antecedents(antecedents)

        # Step 2
        consequent = self.fuzzy_sets[self.rules[rule].get_consequent()]
        projected_matrix = self.cylindrical_projection(consequent, antecedent_matrix, vector_lengths)
        print(projected_matrix)
    
    # Step 1
    def combine_antecedents(self, antecedents :[]) -> ():
        num_antecedents = len(antecedents)
        if num_antecedents == 1:
            return self.fuzzy_sets[antecedents[0]], [len(antecedents[0])]
        else:
            lengths = []
            for i in range(len(antecedents)):
                lengths.append(len(self.fuzzy_sets[antecedents[i]]))
            antecedent_matrix = np.zeros(lengths, dtype=float)

            if num_antecedents == 2:
                for i, ant_i in enumerate(self.fuzzy_sets[antecedents[0]]):
                    for j, ant_j in enumerate(self.fuzzy_sets[antecedents[1]]):
                        antecedent_matrix[i][j] = min(ant_i, ant_j)
            elif num_antecedents == 3:
                for i, ant_i in enumerate(self.fuzzy_sets[antecedents[0]]):
                    for j, ant_j in enumerate(self.fuzzy_sets[antecedents[1]]):
                        for k, ant_k in enumerate(self.fuzzy_sets[antecedents[2]]):
                            antecedent_matrix[i][j][k] = min(ant_i, ant_j, ant_k)
        
            return antecedent_matrix, lengths
        
    # Step 2 - Cylindrical Projection onto consequent
    def cylindrical_projection(self, consequent :[], antecedent_matrix :[], vector_lengths :[]) -> []:
        vector_lengths.append(len(consequent))
        projected_matrix = np.zeros(vector_lengths, dtype=float)

        if len(vector_lengths) - 1 == 1:
            for i, ant in enumerate(antecedent_matrix):
                for j, cons in enumerate(consequent):
                    projected_matrix[i][j] = min(ant, cons)
        elif len(vector_lengths) - 1 == 2:
            for i in range(vector_lengths[0]):
                for j in (vector_lengths[1]):
                    for k, cons in enumerate(consequent):
                        projected_matrix[i][j][k] = min(antecedent_matrix[i][j], cons)
        elif len(vector_lengths) - 1 == 3:
            for i in range(vector_lengths[0]):
                for j in (vector_lengths[1]):
                    for k in (vector_lengths[2]):
                        for l, cons in enumerate(consequent):
                            projected_matrix[i][j][k][l] = min(antecedent_matrix[i][j][k], cons)

        return projected_matrix

    

    # Helpers
    # ----------------------------------------------------------------------
    def perform_implication_operator(self, antecedents :[], consequent :float) -> float:
        if self.implication_operator == "Lukasiewicz":
            result = np.minimum(1, (1.0 - sum(antecedents) + consequent))
        elif self.implication_operator == "Correlation_Min":
            result = np.minimum(antecedents, consequent)
        elif self.implication_operator == "Correlation_Product":
            result = np.prod(antecedents) * consequent
        else:
            print(f'{self.implication_operator} Implication Operator is not supported.')
            exit(0)

        return result
