# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 16:16:31 2018

@author: Zoltan
"""

import pystan
from numpy import loadtxt
import matplotlib.pyplot as plt

fitting_code = """
data {
    int<lower=0> N; //Number of samples
    vector[N] z; //redshift
    vector[N] mu; //distance modulus
}
parameters {
    real OmegaM;
    real h;
}
model {
    real s;
    real s2;
    real s3;
    real s4;
    real a;
    real H0;
    real c;
    real eta1;
    real eta2;
    s = ((1-OmegaM)/OmegaM)^(1./3.);
    s2 = s*s;
    s3 = s*s*s;
    s4 = s*s*s*s;
    a = 1/(1+z);
    H0 = 100*h;
    c = 299792458.0;
    eta1 = 2.*sqrt(s*s*s+1.)*(1-0.1540*s+0.4304*s2+0.19097*s3+0.066941*s4)^(-1./8.);
    eta2 = 2.*sqrt(s*s*s+1.)*(a**(-4)-0.1540*s/(a**3)+0.4304*s2/(a**2)+0.19097*s3/a+0.066941*s4)^(-1./8.);
    DL ~ normal(c/H0 * (1+z) * (eta1 - eta2));
    mu ~ normal(25 - 5*log10(h) + 5*log10(DL));
}
"""
    
data = loadtxt("jla_mub_0.txt")
cov_matrix = loadtxt("jla_mub_covmatrix.txt")
data_z = []
data_mu = []
for i in data:
    data_z.append(i[0])
    data_mu.append(i[1])

data_stan = {'N': len(data),
              'z': data_z,
              'mu': data_mu}

sm = pystan.StanModel(model_code=fitting_code)
fit = sm.sampling(data=data_stan, iter=1000, chains=4, n_jobs=1)