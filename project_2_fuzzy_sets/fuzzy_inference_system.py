# Imports
import numpy as np

class FuzzyInferenceSystem:
    membership_functions = []

    def __init__(self, domain_start :float, domain_end :float, domain_step :float) -> None: 
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.domain_step = domain_step

        self.domain = [x for x in range(domain_start, domain_end, domain_step)]

    def create_trapezoid_mf(self, a :float, b :float, c :float, d :float) -> None:
        mf = []
        slope_ab = 1 / (b - a)
        y_int_ab = 1 - (slope_ab * b)
        slope_cd = 1 / (c - d)
        y_int_cd = 1 - (slope_cd * c)
        for i in self.domain:
            if i <= a:
                mf.append(0)
            elif i > a and i < b:
                val = (slope_ab * i) + y_int_ab
                mf.append(val)
            elif i >= b and i <= c:
                mf.append(1)
            else:
                val = (slope_cd * i) + y_int_cd
                mf.append(val)

        self.membership_functions.append(mf)

    
