class Zadeh_Fuzzy_Set:

    def __init__(self, name :str, domain :str, fuzzy_set :[]):
        self.name = name
        self.domain = domain
        self.fuzzy_set = fuzzy_set

    def get_name(self) -> str:
        return self.name
    
    def get_domain(self) -> str:
        return self.domain
    
    def get_fuzzy_set(self) -> []:
        return self.fuzzy_set

