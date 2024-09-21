
import numpy as np

def f(M, beta, theta):
    return np.tan(theta) - 2 / np.tan(beta) * (M**2 * np.sin(beta)**2 - 1) / (M**2 * (1.4 + np.cos(2*beta)) + 2)

def bisection(M, theta, a, b, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        c = (a + b) / 2
        if f(M, c, theta) == 0 or (b - a) / 2 < tol:
            return c
        if f(M, c, theta) * f(M, a, theta) > 0:
            a = c
        else:
            b = c
    return (a + b) / 2

def secant(M, theta, x0, x1, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        fx0 = f(M, x0, theta)
        fx1 = f(M, x1, theta)
        if abs(fx1) < tol:
            return x1
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        x0, x1 = x1, x2
    return x1

def shock_relations(M1, beta):
    M1n = M1 * np.sin(beta)
    M2n = np.sqrt((1 + 0.2 * M1n**2) / (1.4 * M1n**2 - 0.2))
    M2 = M2n / np.sin(beta - np.arctan(2 / np.tan(beta) * (M1**2 * np.sin(beta)**2 - 1) / (M1**2 * (1.4 + np.cos(2*beta)) + 2)))
    p2_p1 = 1 + 1.4 * (M1n**2 - 1)
    rho2_rho1 = (1.4 * M1n**2) / (0.4 * M1n**2 + 1)
    T2_T1 = p2_p1 / rho2_rho1
    return M2, p2_p1, rho2_rho1, T2_T1

M1 = 2.0
theta = np.radians(15)

beta_weak = bisection(M1, theta, 0, np.pi/2)
beta_strong = secant(M1, theta, np.pi/4, np.pi/2)

M2_weak, p2_p1_weak, rho2_rho1_weak, T2_T1_weak = shock_relations(M1, beta_weak)
M2_strong, p2_p1_strong, rho2_rho1_strong, T2_T1_strong = shock_relations(M1, beta_strong)

print(f"Weak shock: beta = {np.degrees(beta_weak):.2f}°")
print(f"M2 = {M2_weak:.2f}, p2/p1 = {p2_p1_weak:.2f}, rho2/rho1 = {rho2_rho1_weak:.2f}, T2/T1 = {T2_T1_weak:.2f}")
print(f"\nStrong shock: beta = {np.degrees(beta_strong):.2f}°")
print(f"M2 = {M2_strong:.2f}, p2/p1 = {p2_p1_strong:.2f}, rho2/rho1 = {rho2_rho1_strong:.2f}, T2/T1 = {T2_T1_strong:.2f}")
