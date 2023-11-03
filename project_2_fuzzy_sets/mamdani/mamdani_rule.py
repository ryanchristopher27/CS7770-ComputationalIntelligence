
class Mamdani_Rule:

    def __init__(self, name :str, input_mfs :[], output_mf :str):
        self.name = name
        self.input_mfs = input_mfs
        self.output_mf = output_mf

    def get_name(self) -> str:
        return self.name

    def get_input_mfs(self) -> []:
        return self.input_mfs
    
    def get_output_mf(self) -> str:
        return self.output_mf