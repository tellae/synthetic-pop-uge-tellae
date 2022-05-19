from itertools import product
from maxentropy import MinDivergenceModel
import math
import numpy as np

ownership = ["tenant", "owner"]
size = ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"]

samplespace = list(product(ownership, size))

print("Samplespace")
print(samplespace)

n = len(samplespace)


def function_prior_prob(x_array):
    return [math.log(1/10) for x in x_array]


def f0(x):
    return x in samplespace


def f1(x):
    return x[0] == "tenant"


def f2(x):
    return x[1] == "1_pers"


def f3(x):
    return x[1] == "2_pers"


def f4(x):
    return x[1] == "3_pers"


def f5(x):
    return x[1] == "4_pers"


f = [f0, f1, f2, f3, f4, f5]

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
