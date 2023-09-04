# Chromosome Class

# Imports

class Chromosome:
    genes = []

    def __init__(self, fitness_score :float) -> None:
        self.fitness_score = fitness_score

    def set_genes(self, genes) -> None:
        self.genes = genes

    def get_genes(self) -> []:
        return self.genes

    def set_fitness_score(self, fitness_score) -> None:
        self.fitness_score = fitness_score

    def get_fitness_score(self) -> float:
        return self.fitness_score