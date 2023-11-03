class Zadeh_Rule:

    def __init__(self, name :str, antecedents :[], consequent :str):
        self.name = name
        self.antecedents = antecedents
        self.consequent = consequent
        self.relation = []

    def get_name(self) -> str:
        return self.name

    def get_antecedents(self) -> []:
        return self.antecedents
    
    def get_consequent(self) -> str:
        return self.consequent
    
    def set_relation(self, relation :[]) -> None:
        self.relation = relation

    def get_relation(self) -> []:
        return self.relation