import numpy as np
#  """
#     Some scripts for calculating seawater properties
    
#     Parameters & Outputs
#     ----------
#     T : float
#         Temperature in degrees C
#     S : float
#         Salinity in ppt
#     k : float
#         Thermal conductivity in [W/m K]
#     nu : float
#         Kinematic viscocity of seawater for S=35
#     rho : float
#         Density [kg/m^3]
#     alpha : float
#         Thermal diffusivity [m^2/s]

# """

def viscocity(T):
    pol=[-1.131311019739306e-11,1.199552027472192e-9,-5.864346822839289e-8,1.828297985908266e-6]
    nu = np.polyval(pol,T)
    return nu

def conductivity(T,S=35):  
    T = 1.00024*T     # convert from T_90 to T_68
    S = S / 1.00472   # convert from S to S_P
    k = 10**(np.log10(240+0.0002*S)+0.434*(2.3-(343.5+0.037*S)/(T+273.15))*(1-(T+273.15)/(647.3+0.03*S))**(1/3)-3)
    return k

def specific_heat(T,S=35):
    T = 1.00024*T      # convert from T_90 to T_68
    S = S / 1.00472    # convert from S to S_P

    A =  4206.8 - 6.6197*S + 1.2288e-2*S**2
    B = -1.1262 + 5.4178e-2*S - 2.2719e-4*S**2
    C =  1.2026e-2 - 5.3566e-4*S + 1.8906e-6*S**2
    D =  6.8777e-7 + 1.517e-6 *S - 4.4268e-9*S**2

    cp = A + B*T + C*T**2 + D*T**3
    return cp

def density(T,S=35):
    
    s = S/1000

    a = [9.9992293295E+02,    
         2.0341179217E-02,    
        -6.1624591598E-03,    
         2.2614664708E-05,    
        -4.6570659168E-08]

    b = [8.0200240891E+02,    
        -2.0005183488E+00,    
         1.6771024982E-02,    
        -3.0600536746E-05,    
        -1.6132224742E-05]

    rho_w = a[0] + a[1]*T + a[2]*T**2 + a[3]*T**3 + a[4]*T**4
    D_rho = b[0]*s + b[1]*s*T + b[2]*s*(T**2) + b[3]*s*(T**3) + b[4]*(s**2)*(T**2)
    rho   = rho_w + D_rho
    return rho

def diffusivity(T,S=35):
    cp = specific_heat(T,S)
    rho = density(T,S)
    k = conductivity(T,S)
    alpha = k/(rho*cp)
    return alpha
