#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 12:34:58 2018

@author: vatsal
"""
import numpy as np
import pystan
from numpy import loadtxt


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
    // Function to calculate the theoretical value of mu given a z
    
    //Calculate a given z 
    real a(real z
    ){
    real a;
    a = 1./ (1.+z); 
    return(a);
    }
    
    //Calculate eta given a and omega_m  
    real eta(real a,
    real OmegaM
    ){
    real s;
    real s2;
    real s3;
    real s4;
    real eta;
    s = ((1.-OmegaM)/OmegaM)^(1./3.);
    s2 = s*s;
    s3 = s*s*s;
    s4 = s*s*s*s;
    eta = 2.*sqrt(s3+1.)*(a^(-4)-0.1540*s/(a^3)+0.4304*s2/(a^2)+0.19097*s3/a+0.066941*s4)^(-1./8.);
    
    return(eta);
     
  }

    //Calculate Dl given z, h ,OmegaM
    real Dl(real z, 
    real h,
    real OmegaM
    ){
    
    
    real H0;
    real c;
    real Dl;
    
    H0 = 100*h;
    c = 299792458.0;
    Dl = c/H0 * (1+z) * ( eta(1.0,OmegaM) - eta( 1./(1.+z), OmegaM));
    
    return(Dl);
    }

    
    //Return mu_th given z, OmegaM and h   
    vector mu_th( vector z,
    real h,
    real OmegaM,
    int N
    ){
    vector[N] mu_th;
    for(i in 1:N){
            mu_th[i] = 25.0 - 5.0*log10(h) + 5*log10(Dl(z[i], 1, OmegaM));
    }
    return(mu_th);
      
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
fit = sm.sampling(data=data_stan, iter=10000, chains=10, n_jobs=-1)












#def likelihood( h , omega_m ):
#    
#    co_v_data = read_covariance()
#    super_data = read_supernovae()
#    
#    temp_sum_ind = []
#    
#    for i in range(len(co_v_data)):
#        for j in range(len(co_v_data)):
#           a =  super_data[i][1]-mu_th( super_data[i][0],omega_m , h ) 
#           b = super_data[j][1]-mu_th( super_data[j][0],omega_m , h )
#           c = 1 / co_v_data[i][j]
#           
#           temp_sum_ind.append( -0.5 * a * b * c ) 
#           
#    return(np.sum(temp_sum_ind))
#    
#           
#    
#    
#    
#    
#
#def mu_th(z,omega_m, h):
#
#    25 - (5*np.log10(h)) +(5*np.log10(D_l(z,omega_m,h)))
#    
#def D_l(z, omega_m, h):
#    H_0 = 100*h
#    (c/H_0) * (1+z) * ( n_func(1, omega_m)  - n_func(1/(1+z), omega_m) ) 
#    
#def n_func(a,omega_m):
#    
#    s = np.cbrt((1-omega_m)/omega_m) 
#    
#    alpha = 2* np.sqrt(1 + (s**3))
#    
#    beta1 = ( 1/ (a**4))
#    beta2 = ( 0.1540*s )/a**3
#    beta3 = (0.4304* (s**2)) / a**2
#    beta4 = 0.19097 * (s**3) / a
#    beta5 = 0.066941 * (s**4)
#    beta = (beta1 - beta2 + beta3 + beta4 +beta5)**(-1/8)
#    
#    return(alpha*beta)

