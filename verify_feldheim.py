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


def integrand(x1,y2,tau,k,n2,params):
    """
    integrand for feldheim formula
    """
    Omega = params['Omega']
    D=params["D"]
    beta=params["beta"]
    g0=params["g0"]

    alpha_val = alpha_func(tau, params)
    Delta_val = Delta_func(x1, tau, params)
    rho_val=rho_func(x1,params)

    Y_val=Y(tau,params)

    arg1=mp.sqrt(Omega)*y2
    part1=hermite_poly(k,arg1)

    arg2=mp.sqrt(Omega)*alpha_val*y2+mp.sqrt(Omega)*Delta_val
    part2=hermite_poly(n2,arg2)

    arg_inside_square=y2-mp.j*Y_val*g0/(D*Omega*(1+alpha_val**2))*mp.sqrt(2*beta)*rho_val\
                    +alpha_val/(1+alpha_val**2)*Delta_val
    arg_for_exp=-half*Omega*(1+alpha_val**2)*arg_inside_square**2
    part3=mp.exp(arg_for_exp)

    val=part1*part2*part3

    return val


def numerical_integral(x1, tau, k, n2, params):
    result = mp.quad(lambda y2: integrand(x1, y2, tau, k, n2, params),
                     [-mp.inf, mp.inf])
    return result


def feldheim(x1, tau, k, n2, params):
    Omega = params['Omega']
    D = params["D"]
    beta = params["beta"]
    g0 = params["g0"]

    alpha_val = alpha_func(tau, params)
    Delta_val = Delta_func(x1, tau, params)
    rho_val = rho_func(x1, params)

    Y_val = Y(tau, params)

    part1=mp.sqrt(2*mp.pi/Omega)

    part2=(alpha_val**2-1)**mp.mpf(k/2)*(1-alpha_val**2)**mp.mpf(n2/2)/(1+alpha_val**2)**mp.mpf((1+k+n2)/2)

    part3=0
    min_k_n2=min(k,n2)
    for R in range(0,min_k_n2+1):
        fac1=mp.factorial(R)*mp.binomial(k,R)*mp.binomial(n2,R)
        fac2=(4*alpha_val/mp.sqrt(-(alpha_val**2-1)**2))**R
        arg_fac3=(mp.j*Y_val*mp.sqrt(2*beta/Omega)*g0/(D*(1+alpha_val**2))*rho_val\
                -mp.sqrt(Omega)*alpha_val/(1+alpha_val**2)*Delta_val)*mp.sqrt((alpha_val**2+1)/(alpha_val**2-1))

        fac3=hermite_poly(k-R,arg_fac3)

        arg_fac4=(mp.j*Y_val*mp.sqrt(2*beta/Omega)*g0*alpha_val/(D*(1+alpha_val**2))*rho_val\
                +mp.sqrt(Omega)*Delta_val/(1+alpha_val**2))*mp.sqrt((alpha_val**2+1)/(1-alpha_val**2))

        fac4=hermite_poly(n2-R,arg_fac4)
        part3+=fac1*fac2*fac3*fac4
    return part1*part2*part3

def sigma_feldheim(x1, tau, k, n2, params):
    Omega = params['Omega']
    D = params["D"]
    beta = params["beta"]
    g0 = params["g0"]

    alpha_val = alpha_func(tau, params)
    Delta_val = Delta_func(x1, tau, params)
    rho_val = rho_func(x1, params)

    Y_val = Y(tau, params)

    part1 = mp.sqrt(2 * mp.pi / Omega)
    sigma=mp.j
    part2=(alpha_val**2-1)**mp.mpf((k+n2)/2)*sigma**n2/((alpha_val**2+1)**mp.mpf((1+k+n2)/2))
    part3 = 0
    min_k_n2 = min(k, n2)
    for R in range(0,min_k_n2+1):
        fac1 = mp.factorial(R) * mp.binomial(k, R) * mp.binomial(n2, R)
        fac2=(4*alpha_val/(sigma*mp.fabs(alpha_val**2-1)))**R
        arg_fac3 = (mp.j * Y_val * mp.sqrt(2 * beta / Omega) * g0 / (D * (1 + alpha_val ** 2)) * rho_val \
                    - mp.sqrt(Omega) * alpha_val / (1 + alpha_val ** 2) * Delta_val) * mp.sqrt((alpha_val ** 2 + 1) / (alpha_val ** 2 - 1))
        fac3 = hermite_poly(k - R, arg_fac3)
        arg_fac4 = (mp.j * Y_val * mp.sqrt(2 * beta / Omega) * g0 * alpha_val / (D * (1 + alpha_val ** 2)) * rho_val \
                    + mp.sqrt(Omega) * Delta_val / (1 + alpha_val ** 2))*1/sigma*mp.sqrt((alpha_val**2+1)/(alpha_val**2-1))
        fac4 = hermite_poly(n2 - R, arg_fac4)
        part3 += fac1 * fac2 * fac3 * fac4
    return part1 * part2 * part3


#######################read from csv
groupNum=int(sys.argv[1])
rowNum=int(sys.argv[2])
inParamFileName="./inParams/inParams"+str(groupNum)+".csv"
print("file name is "+inParamFileName)
dfstr=pd.read_csv(inParamFileName)
oneRow=dfstr.iloc[rowNum,:]
j1H=int(oneRow.loc["j1H"])
j2H=int(oneRow.loc["j2H"])
g0 = mp.mpf(oneRow.loc["g0"])
omega_m = mp.mpf(oneRow.loc["omegam"])  # Now reads full precision string into mpmath
omega_p = mp.mpf(oneRow.loc["omegap"])
omega_c = mp.mpf(oneRow.loc["omegac"])
er = mp.mpf(oneRow.loc["er"])
thetaCoef = mp.mpf(oneRow.loc["thetaCoef"])
theta = thetaCoef * mp.pi
N1 = int(oneRow.loc["N1"])
N2 = int(oneRow.loc["N2"])
tTot = mp.mpf(oneRow.loc["tTot"])
Q = int(oneRow.loc["Q"])
print("j1H="+str(j1H)+", j2H="+str(j2H)+", g0="+str(g0) \
      +", omega_m="+str(omega_m)+", omega_p="+str(omega_p) \
      +", omega_c="+str(omega_c)+", er="+str(er)+", thetaCoef="+str(thetaCoef)+f", N1={N1}, N2={N2}, tTot={tTot}, Q={Q}")


# derived quantities
# 1. Delta_m
Delta_m = omega_m - omega_p

#2. r
r=mp.log(er)
#3. e^{2r}
e2r=er**2
# #4. dt
dt=tTot/mp.mpf(Q)
tau=dt

#5.
lmd=(e2r-1.0/e2r)/(e2r+1.0/e2r)*Delta_m
#6. D
D=(lmd*mp.sin(theta))**2+omega_p**2
#7. mu
mu=lmd*mp.cos(theta)+Delta_m
#8. beta
beta=Delta_m-lmd*mp.cos(theta)
#9. Omega
Omega=mp.sqrt(beta*mu)
print("\n" + "="*80)
print(f"{'DERIVED QUANTITY':<12}")
print("-" * 80)
##Derived Quantities
print(f"{'theta':<12} | {theta}")
print(f"{'Delta_m':<12} | {Delta_m}")
print(f"{'r':<12} | {r}")
print(f"{'lmd':<12} | {lmd}")
print(f"{'mu':<12} | {mu}")
print(f"{'beta':<12} | {beta}")
print(f"{'Omega':<12} | {Omega}")
print(f"{'D':<12} | {D}")
print(f"{'tau':<12} | {tau}")
print("="*80 + "\n")
params = {
    'omega_c': omega_c,
    'omega_m': omega_m,
    'omega_p': omega_p,
    'Delta_m': Delta_m,
    'lmd': lmd,
    'theta': theta,
    'g0': g0,
    'mu': mu,
    'beta': beta,
    'Omega': Omega,
    'D': D
}
half=mp.mpf(0.5)
x1=mp.mpf(0.2)
k=2
n2=1

rst=numerical_integral(x1,tau,k,n2,params)
fh_rst=sigma_feldheim(x1,tau,k,n2,params)
print(rst)
print(fh_rst)
