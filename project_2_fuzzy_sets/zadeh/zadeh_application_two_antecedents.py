#Imports
from zadeh_fis import Zadeh_FIS
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    zadeh_application_two_antecedents()


def zadeh_application_two_antecedents() -> None:
    X1 = [1, 2, 3, 4]
    X2 = ['@', '#', '&']
    Y = ['a', 'b', 'c', 'd', 'e']

    fis = Zadeh_FIS()

    # Set Implication Operator
        # Lukasiewicz, Correlation_Min, Correlation_Product
    fis.set_implication_operator("Lukasiewicz")
    # fis.set_implication_operator("Correlation_Min")
    # fis.set_implication_operator("Correlation_Product")

    # Create Domains
    fis.create_domain(name="X1", domain=X1)
    fis.create_domain(name="X2", domain=X2)
    fis.create_domain(name="Y", domain=Y)

    # Create Fuzzy Sets
    A1 = [1.0, 0.8, 0.0, 0.0]
    A2 = [0.0, 0.6, 1.0]
    B = [0.0, 0.5, 1.0, 0.5, 0.0]

    fis.create_fuzzy_set(name="A1", domain_involved="X1", fuzzy_set=A1)
    fis.create_fuzzy_set(name="A2", domain_involved="X2", fuzzy_set=A2)
    fis.create_fuzzy_set(name="B", domain_involved="Y", fuzzy_set=B)

    # Create Rules
    fis.create_rule(name='R1', antecedents=['A1', 'A2'], consequent='B')

    # Evaluate
    fis.evaluate_zadeh(rule='R1', antecedents=['A1', 'A2'])

    # Defuzzify
        # Aggregation Methods: Sum, Max
        # Returns (output, relation_val)
    output, relation_val = fis.defuzzify('Y', 'Sum')

    print(f'Output: {output}, Relational Value: {relation_val}')

if __name__=="__main__":
    main()