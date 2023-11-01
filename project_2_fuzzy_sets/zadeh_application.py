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

    fis = Zadeh_FIS()

    # Set Implication Operator
        # Lukasiewicz, Correlation_Min, Correlation_Product
    fis.set_implication_operator("Lukasiewicz")

    # Create Domains
    fis.create_domain(name="X", domain=X)
    fis.create_domain(name="Y", domain=Y)

    # Create Fuzzy Sets
    Small = [1.0, 0.8, 0.0, 0.0]
    Medium = [0.0, 0.5, 1.0, 0.5, 0.0]
    Large = [0.0, 0.0, 0.8, 1.0]
    fis.create_fuzzy_set(name="Small", domain_involved="X", set=Small)
    fis.create_fuzzy_set(name="Medium", domain_involved="Y", set=Medium)
    fis.create_fuzzy_set(name="Large", domain_involved="X", set=Large)

    # Create Rules
    fis.create_rule(name='R1', antecedents=['Small'], consequent='Medium')

    # Evaluate
    fis.evaluate_zadeh(rule='R1', antecedents=['Small'])


if __name__=="__main__":
    main()