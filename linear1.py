# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:29:53 2018

@author: Zoltan
"""

import pystan
import random as rnd

linear_code = """
data {
    int<lower=0> N; //Number of samples
    vector[N] x; //x values
    vector[N] y; //y values
}
parameters {
    real alpha; //y intercept
    real beta; //slope
    real sigma; //noise
}
model {
    y ~ normal(alpha + beta * x, sigma); //linear function equation
}
"""

N = 1000
xs = []
ys = []
for i in range(N):
    xs.append(i)
    ys.append((2*i+rnd.gauss(0,0.1)+2)) # 2*x + 2 + noise


linear_dat = {'N': N,
              'x': xs,
              'y': ys}

sm = pystan.StanModel(model_code=linear_code)
fit = sm.sampling(data=linear_dat, iter=1000, chains=4, n_jobs=1)