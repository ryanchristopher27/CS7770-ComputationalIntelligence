# Imports
import numpy as np
import matplotlib.pyplot as plt
from mamdani_rule import Mamdani_Rule
from mamdani_membership_function import Mamdani_MF

class FuzzyInferenceSystem:
    # membership_functions = {}

    def __init__(self):
        self.membership_functions = {}
        self.domains = {}
        self.domain_steps = {}
        self.output_membership_functions = {}
        # self.output_membership_function = []
        self.rules = []

    # Domains
    # ----------------------------------------------------------------------
    def create_domain(self, name: str, domain_start :float, domain_end :float, domain_step :float) -> None:
        domain = np.arange(domain_start, domain_end, domain_step)
        self.domains[name] = domain
        self.domain_steps[name] = domain_step

    # Membership Functions
    # ----------------------------------------------------------------------
    def create_trapezoid_mf(self, domain :str, name :str, a :float, b :float, c :float, d :float, io :str) -> None:
        mf = []
        if a == b:
            slope_ab = 0
        else:
            slope_ab = 1 / (b - a)
        y_int_ab = 1 - (slope_ab * b)
        if c == d:
            slope_cd = 0
        else:
            slope_cd = 1 / (c - d)
        y_int_cd = 1 - (slope_cd * c)
        for i in self.domains[domain]:
            if i > a and i < b:
                val = (slope_ab * i) + y_int_ab
                mf.append(val)
            elif i >= b and i <= c:
                mf.append(1)
            elif i > c and i < d:
                val = (slope_cd * i) + y_int_cd
                mf.append(val)
            else:
                mf.append(0)

        if io != "o":
            self.membership_functions[name] = Mamdani_MF(name, mf, domain)
            # self.membership_functions[name] = mf
        else:
            self.output_membership_functions[name] = Mamdani_MF(name, mf, domain)
            # self.output_membership_functions[name] = mf

    def create_triangle_mf(self, domain: str, name :str, a :float, b :float, c :float, io :str) -> None:
        self.create_trapezoid_mf(domain, name, a, b, b, c, io)

    def create_gaussian_mf(self, domain :str, name :str, mu :float, sigma :float, io :str) -> None:
        mf = []
        A = 1
        for i in self.domains[domain]:
            val = A * np.exp(-(i - mu) ** 2 / (2 * sigma ** 2))
            mf.append(val)

        if io != "o":
            self.membership_functions[name] = Mamdani_MF(name, mf, domain)
            # self.membership_functions[name] = mf
        else:
            self.output_membership_functions[name] = Mamdani_MF(name, mf, domain)
            # self.output_membership_functions[name] = mf

    # Rules
    # ----------------------------------------------------------------------
    def create_rule(self, name :str, input_mfs :[str], output_mf :str) -> None:
        self.rules.append(Mamdani_Rule(name, input_mfs, output_mf))
        
    # Evalutation
    # ----------------------------------------------------------------------
    def evaluate_mamdani(self, data :{}) -> {}:
        evaluation = {}
        for rule in self.rules:
            min_val = 1
            for mf in rule.get_input_mfs():
                
                mf_val = self.membership_functions[mf].get_mf()[self.get_mf_index(mf, data)]
                # Update Min
                if mf_val < min_val:
                    min_val = mf_val


            evaluation[rule.get_name()] = [min(x, min_val) for x in self.output_membership_functions[rule.get_output_mf()].get_mf()]
            # evaluation[rule.get_name()] = [min(x, min_val) for x in self.output_membership_function]

        # for key in self.membership_functions:
        #     evaluation[key] = self.membership_functions[key][self.get_mf_index(x)]

        return evaluation
    
    def defuzzification_mamdani(self, evals :[]) -> []:
        classification = []
        domain_combos = {}
        for i, rules in enumerate(evals):
            for domain in self.domains:
                domain_combos[domain] = combo = [0 for x in range(len(self.domains["Output"]))]
            # combo = [0 for x in range(len(self.domains["Output"]))]

            for key in rules:
                output_domain = self.membership_functions[self.rules[key].get_output_mf()].get_domain()
                domain_combos[output_domain] = np.add(domain_combos[output_domain], rules[key])
                # combo = np.add(combo, rules[key])

            # if sum(combo) == 0:
            #     print('check')

            cl = classify_iris(self.centroid_calc(combo))
            classification.append(cl)

        return classification

    # Plots
    # ----------------------------------------------------------------------
    def plot_membership_functions(self, domain :str) -> None:
        plt.figure(figsize=(8, 5))
        colors = ['skyblue', 'red', 'green', 'yellow', 'orange']
        color_i = 0
        for mf in self.membership_functions:
            # if domain in mf:
            if domain == self.membership_functions[mf].get_domain():
                plt.plot(self.domains[domain], self.membership_functions[mf].get_mf(), 'k')
                plt.fill_between(self.domains[domain], self.membership_functions[mf].get_mf(), color=colors[color_i], alpha=0.4)
                color_i += 1

        plt.title(f"{domain} Plot")
        plt.ylabel('Fuzzy membership')
        plt.xlabel('The domain of interest')
        plt.ylim(-0.1, 1.1)
        plt.show()
    

    # Helpers
    # ----------------------------------------------------------------------
    def get_mf_index(self, mf :str, data :{}) -> int:
        for key in data:
            if key in mf:
                val = data[key]


        return int(val * self.domain_steps[self.membership_functions[mf].get_domain()] * 100)
        # return int(val * 10)
    
    def centroid_calc(self, combo :[]) -> float:
        # mult = np.dot(combo, self.domains["Output"])
        # numerator = sum(np.dot(combo, self.domains["Output"]))
        numerator = sum([combo[i] * x for i, x in enumerate(self.domains["Output"])])
        denominator = sum(combo)

        # if denominator == 0:
        #     return float(0)
        
        centroid = numerator/denominator

        return centroid
    
def classify_iris(val :float) -> int:
    if val <= 33.3:
        return 0
    elif val > 33.3 and val <= 66.6:
        return 1
    else:
        return 2
