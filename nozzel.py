import numpy as np
import matplotlib.pyplot as plt

def nozzle_area(x, L):
    """Define the nozzle area as a function of x."""
    if x < 0.5 * L:
        return 1.0 - 0.5 * (x / (0.5 * L))**2
    else:
        return 0.5 + 0.5 * ((x - 0.5 * L) / (0.5 * L))**2

def maccormack_solver(nx, nt, L, CFL, gamma):
    
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    A = np.array([nozzle_area(xi, L) for xi in x])
    
    rho = np.ones(nx)
    u = np.ones(nx) * 0.1
    p = np.ones(nx)
    
    for n in range(nt):
        dt = CFL * dx / np.max(u + np.sqrt(gamma * p / rho))
        
        rho_p = np.zeros(nx)
        u_p = np.zeros(nx)
        p_p = np.zeros(nx)
        
        for i in range(1, nx-1):
            F1 = rho[i] * u[i] * A[i]
            F2 = (rho[i] * u[i]**2 + p[i]) * A[i]
            F3 = (gamma / (gamma - 1) * p[i] + 0.5 * rho[i] * u[i]**2) * rho[i] * u[i] * A[i]
            
            F1_plus = rho[i+1] * u[i+1] * A[i+1]
            F2_plus = (rho[i+1] * u[i+1]**2 + p[i+1]) * A[i+1]
            F3_plus = (gamma / (gamma - 1) * p[i+1] + 0.5 * rho[i+1] * u[i+1]**2) * rho[i+1] * u[i+1] * A[i+1]
            
            rho_p[i] = rho[i] - dt / dx * (F1_plus - F1)
            u_p[i] = (rho[i] * u[i] - dt / dx * (F2_plus - F2)) / rho_p[i]
            p_p[i] = (gamma - 1) * (rho_p[i] * (gamma / (gamma - 1) * p[i] / rho[i] + 0.5 * u[i]**2) - 
                                    dt / dx * (F3_plus - F3) - 0.5 * rho_p[i] * u_p[i]**2)
        
        for i in range(1, nx-1):
            F1 = rho_p[i] * u_p[i] * A[i]
            F2 = (rho_p[i] * u_p[i]**2 + p_p[i]) * A[i]
            F3 = (gamma / (gamma - 1) * p_p[i] + 0.5 * rho_p[i] * u_p[i]**2) * rho_p[i] * u_p[i] * A[i]
            
            F1_minus = rho_p[i-1] * u_p[i-1] * A[i-1]
            F2_minus = (rho_p[i-1] * u_p[i-1]**2 + p_p[i-1]) * A[i-1]
            F3_minus = (gamma / (gamma - 1) * p_p[i-1] + 0.5 * rho_p[i-1] * u_p[i-1]**2) * rho_p[i-1] * u_p[i-1] * A[i-1]
            
            rho[i] = 0.5 * (rho[i] + rho_p[i] - dt / dx * (F1 - F1_minus))
            u[i] = 0.5 * (u[i] + (rho_p[i] * u_p[i] - dt / dx * (F2 - F2_minus)) / rho[i])
            p[i] = 0.5 * (p[i] + (gamma - 1) * (rho[i] * (gamma / (gamma - 1) * p_p[i] / rho_p[i] + 0.5 * u_p[i]**2) - 
                                                dt / dx * (F3 - F3_minus) - 0.5 * rho[i] * u[i]**2))
        
        rho[0], u[0], p[0] = 1.0, 0.1, 1.0  # Subsonic inlet
        rho[-1], u[-1], p[-1] = 2 * rho[-2] - rho[-3], 2 * u[-2] - u[-3], 2 * p[-2] - p[-3]  # Extrapolation at outlet
    
    return x, rho, u, p

nx = 101
nt = 5000
L = 3.0
CFL = 0.5
gamma = 1.4

x, rho, u, p = maccormack_solver(nx, nt, L, CFL, gamma)

M = u / np.sqrt(gamma * p / rho)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(x, rho)
plt.title('Density')
plt.xlabel('x')
plt.ylabel('ρ')

plt.subplot(2, 2, 2)
plt.plot(x, u)
plt.title('Velocity')
plt.xlabel('x')
plt.ylabel('u')

plt.subplot(2, 2, 3)
plt.plot(x, p)
plt.title('Pressure')
plt.xlabel('x')
plt.ylabel('p')

plt.subplot(2, 2, 4)
plt.plot(x, M)
plt.title('Mach Number')
plt.xlabel('x')
plt.ylabel('M')

plt.tight_layout()
plt.show()
