from fuzzy_inference_system import FuzzyInferenceSystem

def main():
    start, end, step = 0, 10.5, 0.1
    fis = FuzzyInferenceSystem(start, end, step)

    fis.create_trapezoid_mf("mf_1", 1, 2, 4, 5)
    # fis.create_trapezoid_mf("mf_2", 4, 5, 6, 8)
    # fis.create_trapezoid_mf("mf_3", 7, 8, 9, 10)

    # fis.create_triangle_mf("mf_7", 1, 2, 4)
    fis.create_triangle_mf("mf_8", 4, 5, 6)
    # fis.create_triangle_mf("mf_9", 7, 8, 9)

    # fis.create_gaussian_mf("mf_4", 2, 1)
    # fis.create_gaussian_mf("mf_5", 4, 1)
    fis.create_gaussian_mf("mf_6", 7, 1)

    fis.plot_membership_functions()

if __name__=="__main__":
    main()