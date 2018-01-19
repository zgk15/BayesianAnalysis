# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:29:53 2018

@author: Zoltan
"""

import pystan
import random as rnd

linear_code = """
data {
    int<lower=0> N;
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real sigma;
}
model {
    y ~ normal(alpha + beta * x, sigma);
}
"""

N = 50
xs = []
ys = []
for i in range(N):
    xs.append(i)
    ys.append((2*i+rnd.gauss(0,0.1)+2))


linear_dat = {'N': N,
               'x': xs,
               'y': ys}

sm = pystan.StanModel(model_code=linear_code)
fit = sm.sampling(data=linear_dat, iter=1000, chains=4, n_jobs=1)