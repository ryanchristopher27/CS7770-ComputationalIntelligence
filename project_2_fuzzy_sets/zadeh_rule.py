class Zadeh_Rule:

    def __init__(self, name :str, antecedents :[], consequent :str):
        self.name = name
        self.antecedents = antecedents
        self.consequent = consequent

    def get_name(self) -> str:
        return self.name

    def get_antecedents(self) -> []:
        return self.antecedents
    
    def get_consequent(self) -> str:
        return self.consequent