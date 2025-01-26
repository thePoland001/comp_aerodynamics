import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, Union

def validate_inputs(M: float, theta: float, gamma: float) -> None:
    if M <= 1:
        raise ValueError("Initial Mach number must be supersonic (M > 1)")
    if theta < 0:
        raise ValueError("Expansion angle must be positive")
    if gamma <= 1:
        raise ValueError("Specific heat ratio must be greater than 1")

def prandtl_meyer_function(M: float, gamma: float) -> float:
    if M <= 1:
        return 0
    return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))) - np.arctan(np.sqrt(M**2 - 1))

def mach_from_prandtl_meyer(nu: float, gamma: float, M_guess: float = 2.0) -> float:
    def func(M):
        return prandtl_meyer_function(M, gamma) - nu
    
    M = fsolve(func, M_guess)[0]
    if M <= 1 or not np.isfinite(M):
        raise ValueError("No valid supersonic solution found")
    return M

def isentropic_relations(M: float, gamma: float) -> Tuple[float, float, float, float]:
    T_ratio = 1 + (gamma - 1) / 2 * M**2
    p_ratio = T_ratio ** (gamma / (gamma - 1))
    rho_ratio = T_ratio ** (1 / (gamma - 1))
    a_ratio = np.sqrt(T_ratio)  # Speed of sound ratio
    return T_ratio, p_ratio, rho_ratio, a_ratio

def prandtl_meyer_expansion(M1: float, theta: float, gamma: float = 1.4, P1: float = 101325, T1: float = 300) -> dict:
    """
    Calculate Prandtl-Meyer expansion properties
    
    Args:
        M1: Initial Mach number
        theta: Expansion angle in degrees
        gamma: Specific heat ratio
        P1: Initial pressure (Pa)
        T1: Initial temperature (K)
    
    Returns:
        Dictionary containing flow properties
    """
    validate_inputs(M1, theta, gamma)
    
    try:
        nu1 = prandtl_meyer_function(M1, gamma)
        nu2 = nu1 + np.radians(theta)
        M2 = mach_from_prandtl_meyer(nu2, gamma)
        
        T1_T0, p1_p0, rho1_rho0, a1_a0 = isentropic_relations(M1, gamma)
        T2_T0, p2_p0, rho2_rho0, a2_a0 = isentropic_relations(M2, gamma)
        
        # Calculate ratios and absolute values
        results = {
            'M2': M2,
            'T2_T1': T2_T0 / T1_T0,
            'p2_p1': p2_p0 / p1_p0,
            'rho2_rho1': rho2_rho0 / rho1_rho0,
            'a2_a1': a2_a0 / a1_a0,
            'T2': T1 * (T2_T0 / T1_T0),
            'P2': P1 * (p2_p0 / p1_p0),
            'V2': M2 * np.sqrt(gamma * 287 * T1 * (T2_T0 / T1_T0))  # Velocity in m/s
        }
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Error in expansion calculation: {str(e)}")


if __name__ == "__main__":
    try:
        M1 = 2.0
        theta = 10.0
        results = prandtl_meyer_expansion(M1, theta)
        
        print(f"Initial Mach number: {M1:.3f}")
        print(f"Expansion angle: {theta:.1f} degrees")
        print(f"Final Mach number: {results['M2']:.3f}")
        print(f"Temperature ratio (T2/T1): {results['T2_T1']:.3f}")
        print(f"Pressure ratio (p2/p1): {results['p2_p1']:.3f}")
        print(f"Density ratio (rho2/rho1): {results['rho2_rho1']:.3f}")
        print(f"Final velocity: {results['V2']:.1f} m/s")
        print(f"Final temperature: {results['T2']:.1f} K")
        print(f"Final pressure: {results['P2']:.1f} Pa")
        
    except Exception as e:
        print(f"Error: {str(e)}")
