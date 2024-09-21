import numpy as np
from scipy.optimize import fsolve

def prandtl_meyer_function(M, gamma):
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))

def mach_from_prandtl_meyer(nu, gamma, M_guess=2.0):
    func = lambda M: prandtl_meyer_function(M, gamma) - nu
    return fsolve(func, M_guess)[0]

def isentropic_relations(M, gamma):
    T_ratio = 1 + (gamma - 1) / 2 * M**2
    p_ratio = T_ratio ** (gamma / (gamma - 1))
    rho_ratio = T_ratio ** (1 / (gamma - 1))
    return T_ratio, p_ratio, rho_ratio

def prandtl_meyer_expansion(M1, theta, gamma=1.4):
    nu1 = prandtl_meyer_function(M1, gamma)
    nu2 = nu1 + np.radians(theta)
    M2 = mach_from_prandtl_meyer(nu2, gamma)
    
    T1_T0, p1_p0, rho1_rho0 = isentropic_relations(M1, gamma)
    T2_T0, p2_p0, rho2_rho0 = isentropic_relations(M2, gamma)
    
    T2_T1 = T2_T0 / T1_T0
    p2_p1 = p2_p0 / p1_p0
    rho2_rho1 = rho2_rho0 / rho1_rho0
    
    return M2, T2_T1, p2_p1, rho2_rho1

M1 = 2.0
theta = 10.0

M2, T2_T1, p2_p1, rho2_rho1 = prandtl_meyer_expansion(M1, theta)

print(f"Initial Mach number: {M1:.3f}")
print(f"Expansion angle: {theta:.1f} degrees")
print(f"Final Mach number: {M2:.3f}")
print(f"Temperature ratio (T2/T1): {T2_T1:.3f}")
print(f"Pressure ratio (p2/p1): {p2_p1:.3f}")
print(f"Density ratio (rho2/rho1): {rho2_rho1:.3f}")
