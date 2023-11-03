#Imports
from zadeh_fis import Zadeh_FIS
from sklearn import datasets, metrics
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def main():
    zadeh_application()

def zadeh_application() -> None:

    X = [1, 2, 3, 4]
    Y = ['a', 'b', 'c', 'd', 'e']
    A = [5, 6, 7]

    fis = Zadeh_FIS()

    # Set Implication Operator
        # Lukasiewicz, Correlation_Min, Correlation_Product
    fis.set_implication_operator("Lukasiewicz")
    # fis.set_implication_operator("Correlation_Min")
    # fis.set_implication_operator("Correlation_Product")

    # Create Domains
    fis.create_domain(name="X", domain=X)
    fis.create_domain(name="Y", domain=Y)
    fis.create_domain(name="A", domain=A)

    # Create Fuzzy Sets
    Small = [1.0, 0.8, 0.0, 0.0]
    Medium = [0.0, 0.5, 1.0, 0.5, 0.0]
    Large = [0.0, 0.0, 0.8, 1.0]
    Large_2 = [0.0, 0.6, 1.0]
    Y_Out = [0.0, 0.5, 1.0]
    Not_Small = [0.0, 0.2, 1.0, 1.0]
    A1 = [0.2, 0.5, 0.8]
    A2 = [0.1, 0.7, 0.9]
    A3 = [0.3, 0.8, 1.0]
    B = [0, 0.5, 1.0]
    fis.create_fuzzy_set(name="Small", domain_involved="X", fuzzy_set=Small)
    fis.create_fuzzy_set(name="Medium", domain_involved="Y", fuzzy_set=Medium)
    fis.create_fuzzy_set(name="Large", domain_involved="X", fuzzy_set=Large)
    fis.create_fuzzy_set(name="Large_2", domain_involved="A", fuzzy_set=Large_2)
    fis.create_fuzzy_set(name="Y_Out", domain_involved="A", fuzzy_set=Y_Out)
    fis.create_fuzzy_set(name="Not Small", domain_involved="X", fuzzy_set=Not_Small)
    fis.create_fuzzy_set(name="A1", domain_involved="A", fuzzy_set=A1)
    fis.create_fuzzy_set(name="A2", domain_involved="A", fuzzy_set=A2)
    fis.create_fuzzy_set(name="A3", domain_involved="A", fuzzy_set=A3)
    fis.create_fuzzy_set(name="B", domain_involved="A", fuzzy_set=B)

    # Create Rules
    # fis.create_rule(name='R1', antecedents=['Small'], consequent='Medium')
    # fis.create_rule(name='R2', antecedents=['Small', 'Large_2'], consequent='Y_Out')
    fis.create_rule(name='R3', antecedents=['A1', 'A2', 'A3'], consequent='B')
    fis.create_rule(name='R4', antecedents=['Small', 'Large_2', 'Medium'], consequent='B')

    # Evaluate
    # fis.evaluate_zadeh(rule='R1', antecedents=['Small'])
    # fis.evaluate_zadeh(rule='R1', antecedents=['Not Small'])
    # fis.evaluate_zadeh(rule='R2', antecedents=['Small', 'Large_2'])
    # fis.evaluate_zadeh(rule='R3', antecedents=['A1', 'A2', 'A3'])
    fis.evaluate_zadeh(rule='R4', antecedents=['Small', 'Large_2', 'Medium'])

    # Defuzzify
        # Aggregation Methods: Sum, Max
        # Returns (output, relation_val)
    # output, relation_val = fis.defuzzify('Y', 'Sum')
    output, relation_val = fis.defuzzify('A', 'Sum')
    # output, relation_val = fis.defuzzify('A', 'Sum')

    print(f'Output: {output}, Relational Value: {relation_val}')

if __name__=="__main__":
    main()