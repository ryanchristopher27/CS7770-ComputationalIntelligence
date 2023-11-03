class Mamdani_MF:

    def __init__(self, name :str, mf :[], domain :str):
        self.name = name
        self.mf = mf
        self.domain = domain
    
    def get_name(self) -> str:
        return self.name
    
    def get_mf(self) -> []:
        return self.mf
    
    def get_domain(self) -> str:
        return self.domain