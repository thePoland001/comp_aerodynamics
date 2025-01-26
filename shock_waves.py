import numpy as np
from typing import Tuple, Optional

class ObliqueShockSolver:
    def __init__(self, gamma: float = 1.4):
        self.gamma = gamma
        
    def theta_beta_mach(self, M1: float, beta: float, theta: float) -> float:
        """Theta-beta-M relation for oblique shock."""
        M1_sin_beta = M1 * np.sin(beta)
        num = 2 * (M1_sin_beta**2 - 1)
        den = M1**2 * (self.gamma + np.cos(2*beta)) + 2
        return np.tan(theta) - 2/np.tan(beta) * num/den
    
    def bisection(self, M1: float, theta: float, beta_min: float, 
                 beta_max: float, tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
        """Find shock angle using bisection method."""
        if M1 < 1:
            raise ValueError("Upstream Mach number must be supersonic")
            
        a, b = beta_min, beta_max
        fa = self.theta_beta_mach(M1, a, theta)
        fb = self.theta_beta_mach(M1, b, theta)
        
        if fa * fb > 0:
            return None
            
        for i in range(max_iter):
            c = (a + b) / 2
            fc = self.theta_beta_mach(M1, c, theta)
            
            if abs(fc) < tol:
                return c
                
            if fc * fa < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
                
            if abs(b - a) < tol:
                return (a + b) / 2
                
        raise RuntimeError("Bisection method failed to converge")
    
    def secant(self, M1: float, theta: float, beta1: float, 
               beta2: float, tol: float = 1e-6, max_iter: int = 100) -> Optional[float]:
        """Find shock angle using secant method."""
        if M1 < 1:
            raise ValueError("Upstream Mach number must be supersonic")
            
        x0, x1 = beta1, beta2
        
        for i in range(max_iter):
            fx0 = self.theta_beta_mach(M1, x0, theta)
            fx1 = self.theta_beta_mach(M1, x1, theta)
            
            if abs(fx1) < tol:
                return x1
                
            try:
                x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            except ZeroDivisionError:
                return None
                
            if abs(x2 - x1) < tol:
                return x2
                
            x0, x1 = x1, x2
            
        raise RuntimeError("Secant method failed to converge")
    
    def shock_relations(self, M1: float, beta: float) -> Tuple[float, float, float, float]:
        """Calculate post-shock conditions."""
        M1n = M1 * np.sin(beta)
        
        if M1n <= 1:
            raise ValueError("Normal component of Mach number must be supersonic")
            
        # Normal shock relations
        M2n = np.sqrt((1 + 0.5*(self.gamma-1)*M1n**2) / 
                     (self.gamma*M1n**2 - 0.5*(self.gamma-1)))
        
        # Deflection angle
        theta = np.arctan(2/np.tan(beta) * 
                         (M1n**2 - 1)/(M1**2*(self.gamma + np.cos(2*beta)) + 2))
        
        # Oblique shock relations
        M2 = M2n/np.sin(beta - theta)
        p2_p1 = 1 + 2*self.gamma/(self.gamma+1)*(M1n**2 - 1)
        rho2_rho1 = ((self.gamma+1)*M1n**2) / (2 + (self.gamma-1)*M1n**2)
        T2_T1 = p2_p1/rho2_rho1
        
        return M2, p2_p1, rho2_rho1, T2_T1
    
    def find_shock_solutions(self, M1: float, theta_deg: float) -> dict:
        """Find both weak and strong shock solutions."""
        theta = np.radians(theta_deg)
        
        # Calculate maximum deflection angle
        beta_max = np.arcsin(1/M1)
        theta_max = self.max_deflection_angle(M1)
        
        if theta > theta_max:
            raise ValueError(f"Deflection angle {theta_deg}° exceeds maximum {np.degrees(theta_max):.2f}°")
        
        # Find weak and strong solutions
        beta_weak = self.bisection(M1, theta, beta_max, np.pi/2)
        beta_strong = self.secant(M1, theta, np.pi/4, np.pi/2)
        
        results = {}
        if beta_weak:
            results['weak'] = {
                'beta': np.degrees(beta_weak),
                **dict(zip(['M2', 'p2_p1', 'rho2_rho1', 'T2_T1'],
                          self.shock_relations(M1, beta_weak)))
            }
        
        if beta_strong:
            results['strong'] = {
                'beta': np.degrees(beta_strong),
                **dict(zip(['M2', 'p2_p1', 'rho2_rho1', 'T2_T1'],
                          self.shock_relations(M1, beta_strong)))
            }
            
        return results
    
    def max_deflection_angle(self, M1: float) -> float:
        """Calculate maximum deflection angle."""
        def theta_function(beta):
            return -self.theta_beta_mach(M1, beta, 0)
            
        beta_range = np.linspace(np.arcsin(1/M1), np.pi/2, 1000)
        theta_range = [np.arctan(2/np.tan(beta) * 
                     (M1**2*np.sin(beta)**2 - 1)/(M1**2*(self.gamma + np.cos(2*beta)) + 2))
                     for beta in beta_range]
        return max(theta_range)

# Example usage
if __name__ == "__main__":
    solver = ObliqueShockSolver()
    
    try:
        M1 = 2.0
        theta = 15.0
        results = solver.find_shock_solutions(M1, theta)
        
        print(f"Upstream Mach number: M1 = {M1:.2f}")
        print(f"Deflection angle: θ = {theta:.2f}°")
        print(f"\nMaximum deflection angle: {np.degrees(solver.max_deflection_angle(M1)):.2f}°")
        
        for solution_type, values in results.items():
            print(f"\n{solution_type.capitalize()} shock solution:")
            print(f"β = {values['beta']:.2f}°")
            print(f"M2 = {values['M2']:.3f}")
            print(f"p2/p1 = {values['p2_p1']:.3f}")
            print(f"ρ2/ρ1 = {values['rho2_rho1']:.3f}")
            print(f"T2/T1 = {values['T2_T1']:.3f}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
