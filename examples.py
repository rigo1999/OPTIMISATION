"""
Exemples d'utilisation des algorithmes d'optimisation
Examples of using optimization algorithms

Ce script démontre l'utilisation des différents algorithmes
d'optimisation avec et sans contraintes.
"""

import numpy as np
from optimization_sans_contrainte import (
    gradient_descent, newton_method, conjugate_gradient, bfgs,
    rosenbrock, grad_rosenbrock, hess_rosenbrock
)
from optimization_avec_contrainte import (
    penalty_method, augmented_lagrangian, projected_gradient_descent,
    Constraint, project_box, sphere, grad_sphere
)


def main():
    print("=" * 70)
    print("EXEMPLES D'OPTIMISATION AVANCÉE")
    print("Advanced Optimization Examples")
    print("=" * 70)
    print()
    
    # =========================================================================
    # PARTIE 1: OPTIMISATION SANS CONTRAINTE
    # =========================================================================
    print("PARTIE 1: OPTIMISATION SANS CONTRAINTE")
    print("PART 1: UNCONSTRAINED OPTIMIZATION")
    print("-" * 70)
    print()
    
    # Exemple 1.1: Fonction de Rosenbrock
    print("Exemple 1.1: Fonction de Rosenbrock")
    print("min f(x,y) = (1-x)^2 + 100(y-x^2)^2")
    print("Solution optimale: x* = (1, 1), f* = 0")
    print()
    
    x0_rosenbrock = np.array([0.0, 0.0])
    
    print("  a) Descente de gradient:")
    x_opt, f_opt, n_iter = gradient_descent(
        rosenbrock, grad_rosenbrock, x0_rosenbrock, 
        learning_rate=0.001, max_iter=10000
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6e}")
    print(f"     Itérations: {n_iter}")
    print()
    
    print("  b) Méthode de Newton:")
    x_opt, f_opt, n_iter = newton_method(
        rosenbrock, grad_rosenbrock, hess_rosenbrock, x0_rosenbrock
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6e}")
    print(f"     Itérations: {n_iter}")
    print()
    
    print("  c) Gradient conjugué:")
    x_opt, f_opt, n_iter = conjugate_gradient(
        rosenbrock, grad_rosenbrock, x0_rosenbrock
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6e}")
    print(f"     Itérations: {n_iter}")
    print()
    
    print("  d) BFGS:")
    x_opt, f_opt, n_iter = bfgs(
        rosenbrock, grad_rosenbrock, x0_rosenbrock
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6e}")
    print(f"     Itérations: {n_iter}")
    print()
    
    # Exemple 1.2: Fonction quadratique
    print("Exemple 1.2: Fonction quadratique")
    print("min f(x,y) = x^2 + 4y^2")
    print("Solution optimale: x* = (0, 0), f* = 0")
    print()
    
    def quadratic(x):
        return x[0]**2 + 4*x[1]**2
    
    def grad_quadratic(x):
        return np.array([2*x[0], 8*x[1]])
    
    x0_quad = np.array([2.0, 2.0])
    
    print("  BFGS:")
    x_opt, f_opt, n_iter = bfgs(quadratic, grad_quadratic, x0_quad)
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6e}")
    print(f"     Itérations: {n_iter}")
    print()
    print()
    
    # =========================================================================
    # PARTIE 2: OPTIMISATION AVEC CONTRAINTES
    # =========================================================================
    print("PARTIE 2: OPTIMISATION AVEC CONTRAINTES")
    print("PART 2: CONSTRAINED OPTIMIZATION")
    print("-" * 70)
    print()
    
    # Exemple 2.1: Contrainte linéaire d'inégalité
    print("Exemple 2.1: Contrainte linéaire d'inégalité")
    print("min f(x,y) = x^2 + y^2")
    print("sujet à: x + y >= 1")
    print("Solution optimale: x* = (0.5, 0.5), f* = 0.5")
    print()
    
    x0_constr = np.array([1.0, 1.0])
    constraint_ineq = Constraint(lambda x: -(x[0] + x[1] - 1), 'ineq')
    
    print("  a) Méthode de pénalité:")
    x_opt, f_opt, n_iter = penalty_method(
        sphere, grad_sphere, x0_constr, [constraint_ineq]
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6f}")
    print(f"     Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (doit être >= 1)")
    print(f"     Itérations: {n_iter}")
    print()
    
    print("  b) Lagrangien augmenté:")
    x_opt, f_opt, n_iter = augmented_lagrangian(
        sphere, grad_sphere, x0_constr, [constraint_ineq]
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6f}")
    print(f"     Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (doit être >= 1)")
    print(f"     Itérations: {n_iter}")
    print()
    
    # Exemple 2.2: Contrainte d'égalité
    print("Exemple 2.2: Contrainte d'égalité")
    print("min f(x,y) = (x-2)^2 + (y-2)^2")
    print("sujet à: x + y = 2")
    print("Solution optimale: x* = (1, 1), f* = 2")
    print()
    
    def f_shifted(x):
        return (x[0] - 2)**2 + (x[1] - 2)**2
    
    def grad_f_shifted(x):
        return np.array([2*(x[0] - 2), 2*(x[1] - 2)])
    
    x0_eq = np.array([0.5, 0.5])
    constraint_eq = Constraint(lambda x: x[0] + x[1] - 2, 'eq')
    
    print("  Méthode de pénalité:")
    x_opt, f_opt, n_iter = penalty_method(
        f_shifted, grad_f_shifted, x0_eq, [constraint_eq]
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6f}")
    print(f"     Contrainte x+y = {x_opt[0] + x_opt[1]:.6f} (doit être = 2)")
    print(f"     Itérations: {n_iter}")
    print()
    
    # Exemple 2.3: Contraintes de bornes (box constraints)
    print("Exemple 2.3: Contraintes de bornes")
    print("min f(x,y) = (x-2)^2 + (y-2)^2")
    print("sujet à: 0 <= x <= 1, 0 <= y <= 1")
    print("Solution optimale: x* = (1, 1), f* = 2")
    print()
    
    x0_box = np.array([0.5, 0.5])
    bounds = [(0.0, 1.0), (0.0, 1.0)]
    
    def project_func(x):
        return project_box(x, bounds)
    
    print("  Gradient projeté:")
    x_opt, f_opt, n_iter = projected_gradient_descent(
        f_shifted, grad_f_shifted, x0_box, project_func, learning_rate=0.1
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6f}")
    print(f"     Itérations: {n_iter}")
    print()
    
    # Exemple 2.4: Contraintes multiples
    print("Exemple 2.4: Contraintes multiples")
    print("min f(x,y) = x^2 + y^2")
    print("sujet à: x + y >= 2, x >= 0.5")
    print()
    
    constraint1 = Constraint(lambda x: -(x[0] + x[1] - 2), 'ineq')
    constraint2 = Constraint(lambda x: -(x[0] - 0.5), 'ineq')
    
    x0_multi = np.array([2.0, 2.0])
    
    print("  Méthode de pénalité:")
    x_opt, f_opt, n_iter = penalty_method(
        sphere, grad_sphere, x0_multi, [constraint1, constraint2]
    )
    print(f"     x* = [{x_opt[0]:.4f}, {x_opt[1]:.4f}]")
    print(f"     f* = {f_opt:.6f}")
    print(f"     Contrainte 1 (x+y): {x_opt[0] + x_opt[1]:.6f} (doit être >= 2)")
    print(f"     Contrainte 2 (x): {x_opt[0]:.6f} (doit être >= 0.5)")
    print(f"     Itérations: {n_iter}")
    print()
    
    print("=" * 70)
    print("FIN DES EXEMPLES / END OF EXAMPLES")
    print("=" * 70)


if __name__ == "__main__":
    main()
