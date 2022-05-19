from itertools import product
from maxentropy import MinDivergenceModel
import math
import numpy as np

variables = ["ownership", "size"]

ownership = ["tenant", "owner"]
size = ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"]


variables_modalities = [ownership, size]

samplespace = list(product(ownership, size))

samplespace = [{variables[i]: x[i] for i in range(len(x)) } for x in samplespace] 

print("Samplespace")
print(samplespace)

n = len(samplespace)


def function_prior_prob(x_array):
    return [math.log(1/10) for x in x_array]


def f0(x):
    return x in samplespace


def new_feature(variable, modality):
    def f(x):
        return x[variable] == modality
    return f


modals = []

def create_features():

    f = [f0]

    for i, modalities in enumerate(variables_modalities):
        for modality in modalities[:-1]:
            modals.append(modality)

            f_m = new_feature(variables[i], modality)

            f.append(f_m)

    return f

f = create_features()


print("Modèle sans a priori")
model = MinDivergenceModel(f, samplespace, vectorized=False, verbose=False)

K = np.array([1., 0.63, 0.53, 0.27, 0.09, 0.07]).reshape(1, 6)

model.fit(K)
model.show_dist()

print("Modèle avec a priori, uniforme")
model_with_apriori = MinDivergenceModel(f, samplespace, vectorized=False,
                                        verbose=False, prior_log_pdf=function_prior_prob)

model_with_apriori.fit(K)
model_with_apriori.show_dist()
