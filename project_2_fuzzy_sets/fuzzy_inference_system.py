# Imports
import numpy as np
import matplotlib.pyplot as plt

class FuzzyInferenceSystem:
    membership_functions = {}

    def __init__(self, domain_start :float, domain_end :float, domain_step :float) -> None: 
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.domain_step = domain_step

        self.domain = np.arange(domain_start, domain_end, domain_step)
        # self.domain = [x for x in range(domain_start, domain_end, domain_step)]

    def create_trapezoid_mf(self, name :str, a :float, b :float, c :float, d :float) -> None:
        mf = []
        slope_ab = 1 / (b - a)
        y_int_ab = 1 - (slope_ab * b)
        slope_cd = 1 / (c - d)
        y_int_cd = 1 - (slope_cd * c)
        for i in self.domain:
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

        self.membership_functions[name] = mf

    def create_triangle_mf(self, name :str, a :float, b :float, c :float) -> None:
        self.create_trapezoid_mf(name, a, b, b, c)

    def create_gaussian_mf(self, name :str, mu :float, sigma :float) -> None:
        mf = []
        A = 1
        for i in self.domain:
            val = A * np.exp(-(i - mu) ** 2 / (2 * sigma ** 2))
            mf.append(val)

        self.membership_functions[name] = mf

    def plot_membership_functions(self) -> None:
        plt.figure(figsize=(8, 5))
        colors = ['skyblue', 'red', 'green', 'yellow', 'orange']
        for i, mf in enumerate(self.membership_functions):
            plt.plot(self.domain, self.membership_functions[mf], 'k')
            plt.fill_between(self.domain, self.membership_functions[mf], color=colors[i], alpha=0.4)

        plt.ylabel('Fuzzy membership')
        plt.xlabel('The domain of interest')
        plt.ylim(-0.1, 1.1)
        plt.show()
    
