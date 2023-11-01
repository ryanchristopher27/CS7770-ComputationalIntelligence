class Zadeh_Inference:
    
    def __init__(self, consequent_domain :str):
        self.consequent_domain = consequent_domain
        self.output_sets = {}
    
    def get_consequent_domain(self) -> str:
        return self.consequent_domain
    
    def add_output_set(self, name :str, output_set :[]) -> None:
        self.output_sets[name] = output_set

    def get_output_sets(self) -> {}:
        return self.output_sets