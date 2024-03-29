#Imports
from zadeh_fis import Zadeh_FIS
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    zadeh_application()
    # zadeh_application_two_antecedents()

def zadeh_application() -> None:

    X = [1, 2, 3, 4]
    Y = ['a', 'b', 'c', 'd', 'e']

    fis = Zadeh_FIS()

    # Set Implication Operator
        # Lukasiewicz, Correlation_Min, Correlation_Product
    # fis.set_implication_operator("Lukasiewicz")
    # fis.set_implication_operator("Correlation_Min")
    fis.set_implication_operator("Correlation_Product")

    # Create Domains
    fis.create_domain(name="X", domain=X)
    fis.create_domain(name="Y", domain=Y)

    # Create Fuzzy Sets
    A = [1.0, 0.8, 0.0, 0.0]
    B = [0.0, 0.5, 1.0, 0.5, 0.0]
    Not_A = [0.0, 0.2, 1.0, 1.0]

    fis.create_fuzzy_set(name="A", domain_involved="X", fuzzy_set=A)
    fis.create_fuzzy_set(name="B", domain_involved="Y", fuzzy_set=B)
    fis.create_fuzzy_set(name="Not_A", domain_involved="X", fuzzy_set=Not_A)

    # Create Rules
    fis.create_rule(name='R1', antecedents=['A'], consequent='B')

    # Evaluate
    # fis.evaluate_zadeh(rule='R1', antecedents=['A'])
    fis.evaluate_zadeh(rule='R1', antecedents=['Not_A'])

    # Defuzzify
        # Aggregation Methods: Sum, Max
        # Returns (output, relation_val)
    output, relation_val = fis.defuzzify('Y', 'Sum')

    print(f'Output: {output}, Relational Value: {relation_val}')


if __name__=="__main__":
    main()