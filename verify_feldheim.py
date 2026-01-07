import mpmath as mp
from datetime import datetime
import sys
import pandas as pd
# Set high precision for mpmath
mp.mp.dps=30

#this script verifies if Feldheim formula is valid for complex arguments

def hermite_poly(n, x):
    """Compute Hermite polynomial H_n(x) using mpmath"""
    return mp.hermite(n, x)

def hermite_function(n, x):
    """Compute Hermite function u_n(x) = (2^n n! sqrt(pi))^(-1/2) exp(-x^2/2) H_n(x)"""
    norm = mp.power(2, n) * mp.factorial(n) * mp.sqrt(mp.pi)
    return mp.exp(-x**2 / 2) * hermite_poly(n, x) / mp.sqrt(norm)


def alpha_func(tau,params):
    lmd = params['lmd']
    theta = params['theta']
    val=mp.exp(lmd*mp.sin(theta)*tau)
    return val

def rho_func(x1,params):
    omega_c = params['omega_c']
    val=omega_c*x1**2-mp.mpf("0.5")
    return val


def delta_func(tau,params):
    beta = params['beta']
    D = params['D']
    omega_p = params['omega_p']
    lmd = params['lmd']
    theta = params['theta']
    g0 = params['g0']
    alpha_val=alpha_func(tau,params)

    part0=-g0*mp.sqrt(2/beta)*lmd*mp.sin(theta)/D*alpha_val*mp.sin(omega_p*tau)

    part1=g0*mp.sqrt(2/beta)*omega_p/D*alpha_val*mp.cos(omega_p*tau)

    part2=-g0*mp.sqrt(2/beta)*omega_p/D

    return part0+part1+part2


def Delta_func(x1,tau,params):
    """Compute Delta(x1, tau) from equation (150)"""




    rho_val=rho_func(x1,params)

    delta_val=delta_func(tau,params)

    val=rho_val*delta_val
    return val

def Y(t,params):
    omega_p = params['omega_p']
    lmd = params['lmd']
    theta = params['theta']

    val=-omega_p*mp.sin(omega_p*t)+lmd*mp.sin(theta)*mp.cos(omega_p*t)\
        -lmd*mp.sin(theta)*mp.exp(lmd*mp.sin(theta)*t)
    return val