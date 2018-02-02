#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:34:58 2018

@author: vatsal
"""
import numpy as np
import pystan
from numpy import loadtxt
import corner as corner


def read_covariance(): 
    
    cov_matrix = loadtxt("jla_mub_covmatrix.txt")  
    data_cov=[]
    for i in cov_matrix:
        data_cov.append(i)  
    
    data_cov = np.array(data_cov)
    data_cov = data_cov.reshape([31,31])

    return data_cov


def read_supernovae(): 
    
    data = loadtxt("jla_mub_0.txt")
    data_z = []
    data_mu = []
    for i in data:
        data_z.append(i[0])
        data_mu.append(i[1])
        

    return data_z, data_mu



fitting_code = """

functions{   
    
    //Calculate eta given a and omega_m  
    real eta(real a,
    real OmegaM
    ){
    real s;
    real s2;
    real s3;
    real s4;
    real eta_var;
    real eta_const;
    real eta1;
    real eta2;
    real eta3;
    real eta4;
    real eta5;
    real eta6;
    s = ((1.-OmegaM)/OmegaM)^(1./3.);
    s2 = s*s;
    s3 = s*s*s;
    s4 = s*s*s*s;
    eta_const = 2.0*sqrt(s3+1.0);
    eta1 = 1/(a^4);
    eta2 = -0.1540*s/(a^3);
    eta3 = 0.4304*s2/(a*a);
    eta4 = 0.19097*s3/a;
    eta5 = 0.066941*s4;
    eta6 = (eta1+eta2+eta3+eta4+eta5)^(-0.125);
    eta_var = eta_const*eta6;
    return(eta_var);   
     
  }

    //Calculate Dl given z, h ,OmegaM
    real Dl(real z, 
    real h,
    real OmegaM
    ){
    
    
    real H0;
    real c;
    real Dl_var;
    
    H0 = 100*h;
    c = 299792.4580;
    Dl_var = c/H0 * (1+z) * ( eta(1.0,OmegaM) - eta( 1./(1.+z), OmegaM));
    
    return(Dl_var);
    }

    
    //Return mu_th given z, OmegaM and h   
    vector mu_th( vector z,
    real h,
    real OmegaM,
    int N
    ){
    vector[N] mu_th_var;
    for(i in 1:N){
            mu_th_var[i] = 25.0 - 5.0*log10(h) + 5*log10(Dl(z[i], 1, OmegaM));
    }
    return(mu_th_var);
      
    }
    
}


data {
    int<lower=0> N; //Number of samples
    vector[N] z; //redshift
    vector[N] mu; //distance modulus
    cov_matrix[N] cov; // Covariance matrix
}
parameters {
    real<lower=0> OmegaM;
    real<lower=0> h;
}
model {
    OmegaM ~ normal(1,0.1);
    h ~ normal(0.738, 0.024);
    mu ~ multi_normal(mu_th(z, h, OmegaM, N), cov );
}
"""
    
data_z, data_mu = read_supernovae()
data_cov = read_covariance()


data_stan = {'N': len(data_z),
              'z': data_z,
              'mu': data_mu,
              'cov': data_cov}

sm = pystan.StanModel(model_code=fitting_code)
fit = sm.sampling(data=data_stan, iter=1000, chains=4, n_jobs=-1)

chains = fit.extract()

a,b = chains['OmegaM'], chains['h']
data =[]
for i in range(len(a)):
    data.append([a[i],b[i]])

figure = corner.corner(data,labels=[r"$\omega $", r"$h$"],quantiles=[0.25, 0.5, 0.75],show_titles=True, title_kwargs={"fontsize": 12})
   







def fake_data(num_samples, OmegaM, h):
    z = []
    mu = []
    for i in range(num_samples):
        z.append(i)
        mu.append(OmegaM*i + h)
    return(z,mu)
        