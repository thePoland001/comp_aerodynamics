import numpy as np
import matplotlib.pyplot as plt

def nozzle_area(x, L):
    """Define the nozzle area distribution."""
    A_throat = 0.5
    A_inlet = A_outlet = 1.0
    x_throat = 0.5 * L
    
    return np.where(x < x_throat,
                   A_inlet - (A_inlet - A_throat) * (x / x_throat)**2,
                   A_throat + (A_outlet - A_throat) * ((x - x_throat) / (L - x_throat))**2)

def calculate_fluxes(rho, u, p, A, gamma):
    """Calculate conservation equation fluxes."""
    F1 = rho * u * A
    F2 = (rho * u**2 + p) * A
    F3 = (gamma / (gamma - 1) * p + 0.5 * rho * u**2) * rho * u * A
    return F1, F2, F3

def apply_boundary_conditions(rho, u, p, p_back):
    """Apply characteristic-based boundary conditions."""
    # Subsonic inlet (fixed total conditions)
    rho[0] = 1.0
    u[0] = 0.1
    p[0] = 1.0
    
    # Subsonic/supersonic outlet (fixed back pressure or extrapolation)
    if u[-1] / np.sqrt(gamma * p[-1] / rho[-1]) < 1:  # Subsonic
        p[-1] = p_back
        rho[-1] = rho[-2] * (p[-1] / p[-2])**(1/gamma)
        u[-1] = u[-2]
    else:  # Supersonic
        rho[-1] = 2 * rho[-2] - rho[-3]
        u[-1] = 2 * u[-2] - u[-3]
        p[-1] = 2 * p[-2] - p[-3]
    
    return rho, u, p

def check_convergence(residuals, tol=1e-6):
    """Check if solution has converged."""
    if len(residuals) > 100:
        recent_residuals = residuals[-100:]
        return np.std(recent_residuals) < tol
    return False

def maccormack_solver(nx, max_iter, L, CFL, gamma, p_back=0.7):
    # Initialize grid and geometry
    x = np.linspace(0, L, nx)
    dx = L / (nx - 1)
    A = nozzle_area(x, L)
    
    # Initialize flow field
    rho = np.ones(nx)
    u = np.ones(nx) * 0.1
    p = np.ones(nx)
    
    # Initialize convergence monitoring
    residuals = []
    
    for n in range(max_iter):
        rho_old = rho.copy()
        
        # Calculate time step
        dt = CFL * dx / np.max(u + np.sqrt(gamma * p / rho))
        
        # Predictor step
        rho_p = np.zeros(nx)
        u_p = np.zeros(nx)
        p_p = np.zeros(nx)
        
        for i in range(1, nx-1):
            F1, F2, F3 = calculate_fluxes(rho[i], u[i], p[i], A[i], gamma)
            F1_plus, F2_plus, F3_plus = calculate_fluxes(rho[i+1], u[i+1], p[i+1], A[i+1], gamma)
            
            rho_p[i] = rho[i] - dt/dx * (F1_plus - F1)
            u_p[i] = (rho[i] * u[i] - dt/dx * (F2_plus - F2)) / rho_p[i]
            e_p = (rho[i] * (gamma/(gamma-1) * p[i]/rho[i] + 0.5*u[i]**2) - 
                   dt/dx * (F3_plus - F3)) / rho_p[i]
            p_p[i] = (gamma - 1) * rho_p[i] * (e_p - 0.5*u_p[i]**2)
        
        # Corrector step
        for i in range(1, nx-1):
            F1, F2, F3 = calculate_fluxes(rho_p[i], u_p[i], p_p[i], A[i], gamma)
            F1_minus, F2_minus, F3_minus = calculate_fluxes(rho_p[i-1], u_p[i-1], p_p[i-1], A[i-1], gamma)
            
            rho[i] = 0.5 * (rho[i] + rho_p[i] - dt/dx * (F1 - F1_minus))
            u[i] = 0.5 * (u[i] + (rho_p[i]*u_p[i] - dt/dx * (F2 - F2_minus)) / rho[i])
            e = 0.5 * (rho[i]*(gamma/(gamma-1)*p[i]/rho[i] + 0.5*u[i]**2) + 
                      rho_p[i]*(gamma/(gamma-1)*p_p[i]/rho_p[i] + 0.5*u_p[i]**2) - 
                      dt/dx * (F3 - F3_minus)) / rho[i]
            p[i] = (gamma - 1) * rho[i] * (e - 0.5*u[i]**2)
        
        # Apply boundary conditions
        rho, u, p = apply_boundary_conditions(rho, u, p, p_back)
        
        # Check for negative pressure/density
        if np.any(p < 0) or np.any(rho < 0):
            print("Solution diverged - negative pressure or density")
            break
        
        # Monitor convergence
        residual = np.max(np.abs(rho - rho_old))
        residuals.append(residual)
        
        if check_convergence(residuals):
            print(f"Solution converged after {n} iterations")
            break
    
    return x, rho, u, p, residuals

# Simulation parameters
nx = 101
max_iter = 10000
L = 3.0
CFL = 0.5
gamma = 1.4
p_back = 0.7

# Run simulation
x, rho, u, p, residuals = maccormack_solver(nx, max_iter, L, CFL, gamma, p_back)

# Calculate Mach number
M = u / np.sqrt(gamma * p / rho)

# Plot results
plt.figure(figsize=(15, 10))

# Flow variables
plt.subplot(2, 3, 1)
plt.plot(x, rho)
plt.title('Density')
plt.xlabel('x')
plt.ylabel('ρ')

plt.subplot(2, 3, 2)
plt.plot(x, u)
plt.title('Velocity')
plt.xlabel('x')
plt.ylabel('u')

plt.subplot(2, 3, 3)
plt.plot(x, p)
plt.title('Pressure')
plt.xlabel('x')
plt.ylabel('p')

plt.subplot(2, 3, 4)
plt.plot(x, M)
plt.title('Mach Number')
plt.xlabel('x')
plt.ylabel('M')

plt.subplot(2, 3, 5)
plt.plot(x, nozzle_area(x, L))
plt.title('Nozzle Area')
plt.xlabel('x')
plt.ylabel('A')

plt.subplot(2, 3, 6)
plt.semilogy(residuals)
plt.title('Convergence History')
plt.xlabel('Iteration')
plt.ylabel('Residual')

plt.tight_layout()
plt.show()
